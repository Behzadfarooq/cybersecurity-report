from app.tools.calculator import CalculatorTool


def test_cagr_calculation_matches_expected_value() -> None:
    tool = CalculatorTool()
    result = tool.cagr(initial=7351, final=17000, years=8)

    assert result.operation == "cagr"
    assert abs(result.value - 0.1104852958) < 1e-8


def test_expression_evaluation_is_deterministic() -> None:
    tool = CalculatorTool()
    result = tool.evaluate_expression("(20 - 5) / 3")

    assert result.operation == "expression"
    assert abs(result.value - 5.0) < 1e-9
