import pytest
import textwrap

from rotab.runtime.dag_generator import DagGenerator
from rotab.ast.template_node import TemplateNode
from rotab.ast.process_node import ProcessNode
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.step_node import MutateStep, TransformStep, StepNode
from rotab.ast.context.validation_context import ValidationContext, VariableInfo


def make_context():
    return ValidationContext(
        available_vars=set(),
        schemas={
            "schema1": VariableInfo(type="dataframe", columns={"age": "int"}),
            "schema2": VariableInfo(type="dataframe", columns={"id": "int"}),
            "schema3": VariableInfo(type="dataframe", columns={"id": "int", "value": "float"}),
        },
        eval_scope={"merge": lambda left, right, on: left},
    )


@pytest.fixture
def generator():
    ctx = make_context()

    proc_a1 = ProcessNode(
        name="proc_a1",
        inputs=[InputNode(name="df1", io_type="csv", path="df1.csv", schema="schema1")],
        steps=[
            MutateStep(
                name="mutate_a1", input_vars=["df1"], output_vars=["tmp_a"], operations=[{"derive": "x = age + 1"}]
            ),
            TransformStep(
                name="transform_a1", input_vars=["tmp_a"], output_vars=["out_a"], expr="merge(tmp_a, tmp_a, on='age')"
            ),
        ],
        outputs=[OutputNode(name="out_a", io_type="csv", path="out_a.csv", schema=None)],
    )

    proc_a2 = ProcessNode(
        name="proc_a2",
        inputs=[InputNode(name="df2", io_type="csv", path="df2.csv", schema="schema2")],
        steps=[
            MutateStep(
                name="mutate_a2", input_vars=["df2"], output_vars=["tmp_a2"], operations=[{"derive": "z = id * 2"}]
            ),
        ],
        outputs=[OutputNode(name="tmp_a2", io_type="csv", path="tmp_a2.csv", schema=None)],
    )

    tpl_a = TemplateNode(name="tpl_a", processes=[proc_a1, proc_a2])

    proc_b1 = ProcessNode(
        name="proc_b1",
        inputs=[InputNode(name="df3", io_type="csv", path="df3.csv", schema="schema3")],
        steps=[
            TransformStep(
                name="transform_b1",
                input_vars=["df3", "tmp_a2"],
                output_vars=["out_b"],
                expr="merge(df3, tmp_a2, on='id')",
            ),
        ],
        outputs=[OutputNode(name="out_b", io_type="csv", path="out_b.csv", schema=None)],
    )

    tpl_b = TemplateNode(name="tpl_b", depends=["tpl_a"], processes=[proc_b1])

    for tpl in [tpl_a, tpl_b]:
        tpl.validate(ctx)

    return DagGenerator([tpl_a, tpl_b])


def test_template_dependency_edges(generator):
    edges = generator.build_template_edges()
    edge_names = {(src.name, dst.name) for src, dst in edges}
    assert ("tpl_a", "tpl_b") in edge_names
    assert len(edges) == 1


def test_full_step_dependency_edges(generator):
    nodes = generator.get_nodes()
    node_names = {n.name for n in nodes}
    assert {
        "df1",
        "df2",
        "df3",
        "mutate_a1",
        "transform_a1",
        "mutate_a2",
        "transform_b1",
        "out_a",
        "tmp_a2",
        "out_b",
    }.issubset(node_names)

    edges = generator.build_step_edges(nodes)
    edge_names = {(a.name, b.name) for a, b in edges}

    print("Edges:", edge_names)

    # I/O → Step, Step → Step, Step → I/O 全て検証
    expected_edges = {
        ("df1", "mutate_a1"),
        ("mutate_a1", "transform_a1"),
        ("transform_a1", "out_a"),
        ("df2", "mutate_a2"),
        ("mutate_a2", "tmp_a2"),
        ("df3", "transform_b1"),
        ("mutate_a2", "transform_b1"),
        ("transform_b1", "out_b"),
    }

    for edge in expected_edges:
        assert edge in edge_names, f"Missing edge: {edge}"

    assert len(edge_names) == len(expected_edges)


def test_get_nodes_filters(generator):
    # 全ノード取得
    all_nodes = generator.get_nodes()
    all_names = {n.name for n in all_nodes}
    assert {
        "tpl_a",
        "tpl_b",
        "proc_a1",
        "proc_a2",
        "proc_b1",
        "df1",
        "df2",
        "df3",
        "mutate_a1",
        "transform_a1",
        "mutate_a2",
        "transform_b1",
        "out_a",
        "tmp_a2",
        "out_b",
    }.issubset(all_names)

    # tpl_a に含まれるすべてのノード
    tpl_a_nodes = generator.get_nodes(template_name="tpl_a")
    tpl_a_names = {n.name for n in tpl_a_nodes}
    assert {"proc_a1", "proc_a2", "df1", "df2", "mutate_a1", "transform_a1", "mutate_a2", "out_a", "tmp_a2"}.issubset(
        tpl_a_names
    )

    # proc_a1 に含まれるノード
    proc_a1_nodes = generator.get_nodes(process_name="proc_a1")
    proc_a1_names = {n.name for n in proc_a1_nodes}
    assert {"df1", "mutate_a1", "transform_a1", "out_a"}.issubset(proc_a1_names)


# Marmaid の生成テスト
expected_mermaid_output = textwrap.dedent(
    """\
graph TB
%% Nodes
%% Template: tpl_a
subgraph T_tpl_a ["tpl_a"]
  %% Process: user_filter
  subgraph P_user_filter ["user_filter"]
    I_tpl_a__user(["[I]user"])
    S_tpl_a__filter_users(["[S]filter_users"])
    O_tpl_a__filtered_users(["[O]filtered_users"])
    I_tpl_a__user --> S_tpl_a__filter_users
    S_tpl_a__filter_users --> O_tpl_a__filtered_users
  end
  %% Process: trans_summary
  subgraph P_trans_summary ["trans_summary"]
    I_tpl_a__transaction(["[I]transaction"])
    S_tpl_a__summarize_transactions(["[S]summarize_transactions"])
    O_tpl_a__filtered_transactions(["[O]filtered_transactions"])
    I_tpl_a__transaction --> S_tpl_a__summarize_transactions
    S_tpl_a__summarize_transactions --> O_tpl_a__filtered_transactions
  end
end
%% Template: tpl_b
subgraph T_tpl_b ["tpl_b"]
  %% Process: transaction_enrichment
  subgraph P_transaction_enrichment ["transaction_enrichment"]
    I_tpl_b__filtered_users(["[I]filtered_users"])
    I_tpl_b__filtered_transactions(["[I]filtered_transactions"])
    S_tpl_b__merge_data(["[S]merge_data"])
    S_tpl_b__enrich(["[S]enrich"])
    O_tpl_b__enriched(["[O]enriched"])
    I_tpl_b__filtered_users --> S_tpl_b__merge_data
    I_tpl_b__filtered_transactions --> S_tpl_b__merge_data
    S_tpl_b__merge_data --> S_tpl_b__enrich
    S_tpl_b__enrich --> O_tpl_b__enriched
  end
  %% Process: dummy_proc
  subgraph P_dummy_proc ["dummy_proc"]
    I_tpl_b__user(["[I]user"])
    S_tpl_b__noop_step(["[S]noop_step"])
    O_tpl_b__noop_out(["[O]noop_out"])
    I_tpl_b__user --> S_tpl_b__noop_step
    S_tpl_b__noop_step --> O_tpl_b__noop_out
  end
end
%% Template Dependencies
T_tpl_a --> T_tpl_b"""
)


def test_generate_mermaid_with_template_namespacing():
    ctx = ValidationContext(
        available_vars=set(),
        schemas={
            "schema_user": VariableInfo(type="dataframe", columns={"user_id": "str", "age": "int"}),
            "schema_trans": VariableInfo(type="dataframe", columns={"user_id": "str", "amount": "int"}),
        },
        eval_scope={"merge": lambda x, y, on: x},
    )

    tpl_a = TemplateNode(
        name="tpl_a",
        processes=[
            ProcessNode(
                name="user_filter",
                inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema="schema_user")],
                steps=[
                    MutateStep(
                        name="filter_users",
                        input_vars=["user"],
                        output_vars=["filtered_users"],
                        operations=[
                            {"filter": "age >= 18"},
                            {"derive": "age_group = age // 10"},
                            {"select": ["user_id", "age_group"]},
                        ],
                    )
                ],
                outputs=[OutputNode(name="filtered_users", io_type="csv", path="filtered_users.csv", schema=None)],
            ),
            ProcessNode(
                name="trans_summary",
                inputs=[InputNode(name="transaction", io_type="csv", path="transaction.csv", schema="schema_trans")],
                steps=[
                    MutateStep(
                        name="summarize_transactions",
                        input_vars=["transaction"],
                        output_vars=["filtered_transactions"],
                        operations=[
                            {"filter": "amount > 1000"},
                            {"derive": "is_high = amount > 10000"},
                            {"select": ["user_id", "amount", "is_high"]},
                        ],
                    )
                ],
                outputs=[
                    OutputNode(
                        name="filtered_transactions", io_type="csv", path="filtered_transactions.csv", schema=None
                    )
                ],
            ),
        ],
    )

    tpl_b = TemplateNode(
        name="tpl_b",
        depends=["tpl_a"],
        processes=[
            ProcessNode(
                name="transaction_enrichment",
                inputs=[
                    InputNode(name="filtered_users", io_type="csv", path="filtered_users.csv", schema=None),
                    InputNode(
                        name="filtered_transactions", io_type="csv", path="filtered_transactions.csv", schema=None
                    ),
                ],
                steps=[
                    TransformStep(
                        name="merge_data",
                        input_vars=["filtered_users", "filtered_transactions"],
                        output_vars=["merged"],
                        expr="merge(filtered_users, filtered_transactions, on='user_id')",
                    ),
                    MutateStep(
                        name="enrich",
                        input_vars=["merged"],
                        output_vars=["enriched"],
                        operations=[{"derive": "status = 'ok'"}, {"select": ["user_id", "status"]}],
                    ),
                ],
                outputs=[OutputNode(name="enriched", io_type="csv", path="enriched.csv", schema=None)],
            ),
            ProcessNode(
                name="dummy_proc",
                inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema="schema_user")],
                steps=[
                    MutateStep(
                        name="noop_step",
                        input_vars=["user"],
                        output_vars=["noop_out"],
                        operations=[{"derive": "dummy = 1"}, {"select": ["dummy"]}],
                    )
                ],
                outputs=[OutputNode(name="noop_out", io_type="csv", path="noop.csv", schema=None)],
            ),
        ],
    )

    tpl_a.validate(ctx)
    tpl_b.validate(ctx)

    generator = DagGenerator([tpl_a, tpl_b])
    actual_output = generator.generate_mermaid()

    assert (
        actual_output.strip() == expected_mermaid_output.strip()
    ), f"Mermaid output mismatch:\n\nExpected:\n{expected_mermaid_output}\n\nActual:\n{actual_output}"
