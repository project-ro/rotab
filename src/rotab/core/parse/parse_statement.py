import re


def is_expression(s: str) -> bool:
    def remove_quoted_parts(text: str) -> str:
        return re.sub(r"""(['"])(?:(?!\1|\\).|\\.)*\1""", " ", text)

    cleaned = remove_quoted_parts(s)

    operator_pattern = r"[+\-*/%]|==|!=|<=|>=|<|>|\b(and|or|not)\b"

    return re.search(operator_pattern, cleaned) is not None


def parse_select_columns(expr: str) -> str:
    """
    'select col1 and col2, col3' のような表現を
    '[col1,col2,col3]' のようにリスト形式で返す。
    """
    expr = expr.strip()

    if not expr.lower().startswith("select "):
        raise ValueError("Not a select expression")

    rest = expr[7:]

    parts = re.split(r"\s*(?:,|and)\s*", rest)
    cleaned = [p.strip() for p in parts if p.strip()]
    return "[" + ",".join(cleaned) + "]"


def parse_statement(expr: str) -> str:
    """
    Parses a simplified natural language-like statement into a standardized function call string.
    """
    expr = expr.encode("utf-8").decode("unicode_escape")
    normalized_expr = re.sub(r"\s+", " ", expr.strip())

    if not normalized_expr:
        raise ValueError("Empty statement")

    # --- 特例: filter は "/" が混じっているので早期分岐 ---
    if normalized_expr.startswith("filter"):
        if "/" not in normalized_expr:
            raise ValueError(f"Invalid filter syntax: {expr}")
        _, content = normalized_expr.split("/", 1)
        content = content.strip()

        # 論理演算子と条件式を交互に抽出する正規表現
        parts = re.findall(r"(\b(?:and|or)\b|\S+(?: \S+)*?(?= \b(?:and|or)\b|$))", content)

        parsed_parts = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.lower() in ["and", "or"]:
                parsed_parts.append(part.lower())
            else:
                if " for " in part:
                    part = re.sub(r"\bvalid\s+for\s+(\w+)", r"valid_name for \1", part)
                    parsed_parts.append(_parse_nested_expression(part))
                else:
                    parsed_part = re.sub(r"\s*([=><!]=?|==)\s*", r"\1", part)
                    parsed_parts.append(parsed_part)

        return " ".join(parsed_parts)

    # --- 特例: add ---
    if normalized_expr.startswith("add "):
        content = normalized_expr[4:]
        if "/" not in content:
            raise ValueError(f"Invalid add syntax: {expr}")

        lhs, rhs = [s.strip() for s in content.split("/", 1)]
        parsed_rhs = _parse_nested_expression(rhs)

        if parsed_rhs != rhs:
            return f"{lhs}={parsed_rhs}"

        if is_expression(rhs):
            rhs = re.sub(r"\s*([+\-*/%()])\s*", r"\1", rhs)
            rhs = re.sub(r"\s+([=><!]=?|==)\s*", r"\1", rhs)
            return f"{lhs}={rhs}"

        return f"{lhs}={rhs}"

    # --- 特例: select ---
    if normalized_expr.lower().startswith("select "):
        return parse_select_columns(normalized_expr)

    # --- 通常関数の解析 ---
    tokens = normalized_expr.split(" ")
    func_name = tokens[0]

    args_str = normalized_expr[len(func_name) :].strip()
    if not args_str:
        return f"{func_name}()"

    args_sections = re.split(r",\s*(?![^()]*\))", args_str)

    parsed_args = []
    for section in args_sections:
        section = section.strip()
        if not section:
            continue

        kv_match = re.match(r"(\w+)\s*(=?)\s*(.*)", section)
        if not kv_match:
            parsed_args.append(section)
            continue

        key, _, value_str = kv_match.groups()
        value_str = value_str.strip()

        if " and " in value_str:
            parts = [p.strip() for p in re.split(r"\s+and\s+", value_str)]

            if all(len(p.split()) == 1 for p in parts):
                parsed_args.append(f"{key}=[{','.join(parts)}]")

            elif all(len(p.split()) == 2 for p in parts):
                kv_pairs = [f"{k}:{v}" for k, v in (p.split() for p in parts)]
                parsed_args.append(f"{key}={{{','.join(kv_pairs)}}}")

            else:
                raise ValueError(f"Inconsistent parts in '{key}': {value_str}")
        else:
            parsed_args.append(f"{key}={value_str}")

    return f"{func_name}({','.join(parsed_args)})"


def _parse_nested_expression(expr: str) -> str:
    """
    Parses a nested expression like 'formatting for yyyymm from '%Y%m' to '%Y-%m''
    into 'formatting(for=yyyymm,from='%Y%m',to='%Y-%m')'
    """
    expr = expr.strip()

    # 数式の場合は変換しない
    if is_expression(expr):
        return expr

    tokens = re.findall(r"(\w+|'.*?'|\".*?\"|\S+)", expr)

    if len(tokens) < 2:
        return expr

    func_name = tokens[0]
    args = tokens[1:]

    result = []
    i = 0
    while i < len(args):
        if i + 1 < len(args):
            result.append(f"{args[i]}={args[i+1]}")
            i += 2
        else:
            result.append(args[i])
            i += 1

    return f"{func_name}({','.join(result)})"
