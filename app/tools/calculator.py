from __future__ import annotations

import ast
import operator
from dataclasses import dataclass


_ALLOWED_BINARY_OPERATORS: dict[type[ast.AST], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARY_OPERATORS: dict[type[ast.AST], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class CalculationResult:
    operation: str
    value: float
    details: dict[str, float | str]


class CalculatorTool:
    name = "calculator"
    description = "Deterministic calculator for expressions and CAGR computations."

    def cagr(self, initial: float, final: float, years: float) -> CalculationResult:
        if initial <= 0 or final <= 0:
            raise ValueError("Initial and final values must be positive for CAGR.")
        if years <= 0:
            raise ValueError("Years must be greater than zero for CAGR.")

        value = (final / initial) ** (1.0 / years) - 1.0
        return CalculationResult(
            operation="cagr",
            value=value,
            details={"initial": initial, "final": final, "years": years},
        )

    def evaluate_expression(self, expression: str) -> CalculationResult:
        parsed = ast.parse(expression, mode="eval")
        value = float(self._eval_node(parsed.body))
        return CalculationResult(operation="expression", value=value, details={"expression": expression})

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return float(_ALLOWED_BINARY_OPERATORS[type(node.op)](left, right))

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
            operand = self._eval_node(node.operand)
            return float(_ALLOWED_UNARY_OPERATORS[type(node.op)](operand))

        raise ValueError("Unsupported expression for safe evaluation.")
