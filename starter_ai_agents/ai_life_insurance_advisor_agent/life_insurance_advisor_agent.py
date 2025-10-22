import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.e2b import E2BTools
from agno.tools.firecrawl import FirecrawlTools

st.set_page_config(
    page_title="人寿保险 coverage 顾问",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ 人寿保险 coverage 顾问")
st.caption(
    "由 Agno Agents、OpenAI GPT-5、E2B 沙盒代码执行和 Firecrawl 搜索提供支持的原型 Streamlit 应用。"
)

# -----------------------------------------------------------------------------
# 用于 API 密钥的侧边栏配置
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("API 密钥")
    st.write("所有密钥仅保存在您的浏览器会话中。")
    openai_api_key = st.text_input(
        "OpenAI API 密钥",
        type="password",
        key="openai_api_key",
        help="在 https://platform.openai.com/api-keys 创建",
    )
    firecrawl_api_key = st.text_input(
        "Firecrawl API 密钥",
        type="password",
        key="firecrawl_api_key",
        help="在 https://www.firecrawl.dev/app/api-keys 创建",
    )
    e2b_api_key = st.text_input(
        "E2B API 密钥",
        type="password",
        key="e2b_api_key",
        help="在 https://e2b.dev 创建",
    )
    st.markdown("---")
    st.caption(
        "该代理使用 E2B 进行确定性 coverage 计算，并使用 Firecrawl 进行最新的定期寿险产品研究。"
    )

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def safe_number(value: Any) -> float:
    """尽最大努力将代理输出转换为浮点数。"""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            stripped = value
            for token in [",", "$", "€", "£", "₹", "C$", "A$"]:
                stripped = stripped.replace(token, "")
            stripped = stripped.strip()
            try:
                return float(stripped)
            except ValueError:
                return 0.0
        return 0.0


def format_currency(amount: float, currency_code: str) -> str:
    symbol_map = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "CAD": "C$",
        "AUD": "A$",
        "INR": "₹",
    }
    code = (currency_code or "USD").upper()
    symbol = symbol_map.get(code, "")
    formatted = f"{amount:,.0f}"
    return f"{symbol}{formatted}" if symbol else f"{formatted} {code}"


def extract_json(payload: str) -> Optional[Dict[str, Any]]:
    if not payload:
        return None

    content = payload.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def parse_percentage(value: Any, fallback: float = 0.02) -> float:
    """将百分比类值转换为小数形式（例如，"2%" -> 0.02）。"""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        # 如果小于 1，则假设已为小数，否则视为百分比值
        return float(value) if value < 1 else float(value) / 100
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        try:
            numeric = float(cleaned)
            return numeric if numeric < 1 else numeric / 100
        except ValueError:
            return fallback
    return fallback


def compute_local_breakdown(profile: Dict[str, Any], real_rate: float) -> Dict[str, float]:
    """在本地复制 coverage 计算，以便向用户展示。"""
    income = safe_number(profile.get("annual_income"))
    years = max(0, int(profile.get("income_replacement_years", 0) or 0))
    total_debt = safe_number(profile.get("total_debt"))
    savings = safe_number(profile.get("available_savings"))
    existing_cover = safe_number(profile.get("existing_life_insurance"))

    if real_rate <= 0:
        discounted_income = income * years
        annuity_factor = years
    else:
        annuity_factor = (1 - (1 + real_rate) ** (-years)) / real_rate if years else 0
        discounted_income = income * annuity_factor

    assets_offset = savings + existing_cover
    recommended = max(0.0, discounted_income + total_debt - assets_offset)

    return {
        "income": income,
        "years": years,
        "real_rate": real_rate,
        "annuity_factor": annuity_factor,
        "discounted_income": discounted_income,
        "debt": total_debt,
        "assets_offset": -assets_offset,
        "recommended": recommended,
    }


@st.cache_resource(show_spinner=False)
def get_agent(openai_key: str, firecrawl_key: str, e2b_key: str) -> Optional[Agent]:
    if not (openai_key and firecrawl_key and e2b_key):
        return None

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
    os.environ["E2B_API_KEY"] = e2b_key

    return Agent(
        name="人寿保险顾问",
        model=OpenAIChat(
            id="gpt-5-mini-2025-08-07",
            api_key=openai_key,
        ),
        tools=[
            E2BTools(timeout=180),
            FirecrawlTools(
                api_key=firecrawl_key,
                enable_search=True,
                enable_crawl=True,
                enable_scrape=False,
                search_params={"limit": 5, "lang": "en"},
            ),
        ],
        instructions=[
            "您提供保守的人寿保险指导。您的工作流程严格如下：",
            "1. 始终从 E2B 工具调用 `run_python_code`，使用提供的客户 JSON 计算 coverage 建议。",
            "   - 将缺失的数值视为 0。",
            "   - 贴现收入替代现金流时，使用 2% 的默认实际贴现率。",
            "   - 计算：贴现收入 = 年收入 * ((1 - (1 + r)^(-收入替代年数)) / r)。",
            "   - 建议 coverage = max(0, 贴现收入 + 总债务 - 储蓄 - 现有人寿保险)。",
            "   - 打印包含以下键的 JSON：coverage_amount、coverage_currency、breakdown、assumptions。",
            "2. 使用 Firecrawl 的 `search` 以及可选的 `scrape_website` 调用，收集客户所在地区最新的定期寿险选项。",
            "3. 仅返回包含以下顶级键的 JSON：coverage_amount、coverage_currency、breakdown、assumptions、recommendations、research_notes、timestamp。",
            "   - `coverage_amount`：总建议 coverage 的整数。",
            "   - `coverage_currency`：3 字母货币代码。",
            "   - `breakdown`：包括 income_replacement、debt_obligations、assets_offset、methodology。",
            "   - `assumptions`：包括 income_replacement_years、real_discount_rate、additional_notes。",
            "   - `recommendations`：最多三个对象的列表（name、summary、link、source）。",
            "   - `research_notes`：简短免责声明 + 来源的时效性。",
            "   - `timestamp`：ISO 8601 日期时间字符串。",
            "最终 JSON 输出中不要包含 markdown、评论或工具调用跟踪。",
        ],
        markdown=False,
    )


# -----------------------------------------------------------------------------
# 用户输入表单
# -----------------------------------------------------------------------------
st.subheader("告诉我们关于您的信息")

with st.form("coverage_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=18, max_value=85, value=35)
        annual_income = st.number_input(
            "年收入",
            min_value=0.0,
            value=85000.0,
            step=1000.0,
        )
        dependents = st.number_input(
            "受抚养人数量",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )
        location = st.text_input(
            "国家/州",
            value="美国",
            help="用于本地化推荐的保险公司。",
        )
    with col2:
        total_debt = st.number_input(
            "未偿还债务总额（包括抵押贷款）",
            min_value=0.0,
            value=200000.0,
            step=5000.0,
        )
        savings = st.number_input(
            "可用于受抚养人的储蓄和投资",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
        )
        existing_cover = st.number_input(
            "现有人寿保险",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
        )
        currency = st.selectbox(
            "货币",
            options=["USD", "CAD", "EUR", "GBP", "AUD", "INR"],
            index=0,
        )

    income_replacement_years = st.selectbox(
        "收入替代期限",
        options=[5, 10, 15],
        index=1,
        help="应为受抚养人替代您收入的年数。",
    )

    submitted = st.form_submit_button("生成 Coverage 和选项")


def build_client_profile() -> Dict[str, Any]:
    return {
        "age": age,
        "annual_income": annual_income,
        "dependents": dependents,
        "location": location,
        "total_debt": total_debt,
        "available_savings": savings,
        "existing_life_insurance": existing_cover,
        "income_replacement_years": income_replacement_years,
        "currency": currency,
        "request_timestamp": datetime.utcnow().isoformat(),
    }


def render_recommendations(result: Dict[str, Any], profile: Dict[str, Any]) -> None:
    coverage_currency = result.get("coverage_currency", currency)
    coverage_amount = safe_number(result.get("coverage_amount", 0))

    st.subheader("建议 Coverage")
    st.metric(
        label="所需总 Coverage",
        value=format_currency(coverage_amount, coverage_currency),
    )

    assumptions = result.get("assumptions", {})
    real_rate = parse_percentage(assumptions.get("real_discount_rate", "2%"))
    local_breakdown = compute_local_breakdown(profile, real_rate)

    st.subheader("计算输入")
    st.table(
        {
            "输入": [
                "年收入",
                "收入替代期限",
                "总债务",
                "流动性资产",
                "现有寿险 coverage",
                "实际贴现率",
            ],
            "值": [
                format_currency(local_breakdown["income"], coverage_currency),
                f"{local_breakdown['years']} 年",
                format_currency(local_breakdown["debt"], coverage_currency),
                format_currency(safe_number(profile.get("available_savings")), coverage_currency),
                format_currency(safe_number(profile.get("existing_life_insurance")), coverage_currency),
                f"{real_rate * 100:.2f}%",
            ],
        }
    )

    st.subheader("分步 Coverage 计算")
    step_rows = [
        ("年金因子", f"{local_breakdown['annuity_factor']:.3f}"),
        ("贴现收入替代", format_currency(local_breakdown["discounted_income"], coverage_currency)),
        ("+ 未偿还债务", format_currency(local_breakdown["debt"], coverage_currency)),
        ("- 资产和现有 coverage", format_currency(local_breakdown["assets_offset"], coverage_currency)),
        ("= 公式估算", format_currency(local_breakdown["recommended"], coverage_currency)),
    ]
    step_rows.append(("= 代理建议", format_currency(coverage_amount, coverage_currency)))

    st.table({"步骤": [s for s, _ in step_rows], "金额": [a for _, a in step_rows]})

    breakdown = result.get("breakdown", {})
    with st.expander("此数字的计算方式", expanded=True):
        st.markdown(
            f"- 收入替代价值：{format_currency(safe_number(breakdown.get('income_replacement')), coverage_currency)}"
        )
        st.markdown(
            f"- 债务义务：{format_currency(safe_number(breakdown.get('debt_obligations')), coverage_currency)}"
        )
        assets_offset = safe_number(breakdown.get("assets_offset"))
        st.markdown(
            f"- 资产和现有 coverage 抵消：{format_currency(assets_offset, coverage_currency)}"
        )
        methodology = breakdown.get("methodology")
        if methodology:
            st.caption(methodology)

    recommendations = result.get("recommendations", [])
    if recommendations:
        st.subheader("顶级定期寿险选项")
        for idx, option in enumerate(recommendations, start=1):
            with st.container():
                name = option.get("name", "未命名产品")
                summary = option.get("summary", "未提供摘要。")
                st.markdown(f"**{idx}. {name}** — {summary}")
                link = option.get("link")
                if link:
                    st.markdown(f"[查看详情]({link})")
                source = option.get("source")
                if source:
                    st.caption(f"来源：{source}")
                st.markdown("---")

    with st.expander("模型假设"):
        st.write(
            {
                "收入替代年数": assumptions.get(
                    "income_replacement_years", income_replacement_years
                ),
                "实际贴现率": assumptions.get("real_discount_rate", "2%"),
                "备注": assumptions.get("additional_notes", ""),
            }
        )

    if result.get("research_notes"):
        st.caption(result["research_notes"])
    if result.get("timestamp"):
        st.caption(f"生成时间：{result['timestamp']}")

    with st.expander("代理响应 JSON"):
        st.json(result)


if submitted:
    if not all([openai_api_key, firecrawl_api_key, e2b_api_key]):
        st.error("请在侧边栏配置 OpenAI、Firecrawl 和 E2B API 密钥。")
        st.stop()

    advisor_agent = get_agent(openai_api_key, firecrawl_api_key, e2b_api_key)
    if not advisor_agent:
        st.error("无法初始化顾问。请仔细检查 API 密钥。")
        st.stop()

    client_profile = build_client_profile()
    user_prompt = (
        "您将收到一个描述客户资料的 JSON 对象。按照您的工作流程说明计算 coverage 并提供合适的产品。\n"
        f"客户资料 JSON：{json.dumps(client_profile)}"
    )

    with st.spinner("咨询顾问代理中..."):
        response = advisor_agent.run(user_prompt, stream=False)

    parsed = extract_json(response.content if response else "")
    if not parsed:
        st.error("代理返回了意外响应。在下方启用调试以检查原始输出。")
        with st.expander("原始代理输出"):
            st.write(response.content if response else "<空>")
    else:
        render_recommendations(parsed, client_profile)
        with st.expander("代理调试"):
            st.write(response.content)

st.divider()
st.caption(
    "此原型仅用于教育目的，不提供有执照的财务建议。"
    "请与合格的专业人士和列出的保险公司核实所有建议。"
)