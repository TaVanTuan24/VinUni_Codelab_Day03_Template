"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

# --- Hằng số hỗ trợ phân tích truy vấn tiếng Việt (Simulated LLM Parser) ---
AIRPORT_CODES = ["HAN", "SGN", "DAD"]
CITY_NAME_TO_CODE = {
    "ha noi": "HAN", "hanoi": "HAN",
    "sai gon": "SGN", "ho chi minh": "SGN",
    "da nang": "DAD", "danang": "DAD",
}
FLIGHT_KEYWORDS = re.compile(r"chuyến bay|vé máy bay|vé bay|đặt vé|sân bay")
WEATHER_KEYWORDS = re.compile(r"thời tiết|weather|mặc gì|nên mặc|trang phục")


def strip_accents(text: str) -> str:
    """Chuẩn hoá tiếng Việt có dấu -> không dấu ('Đà Nẵng' -> 'da nang')."""
    nfkd = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).replace("đ", "d")


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool).
        # Baseline KHÔNG có quyền truy cập tool / database -> tool_calls luôn rỗng.
        return {
            "status": "success",
            "answer": (
                "[Chatbot Baseline] Tôi không thể tra cứu dữ liệu thực tế "
                "(chuyến bay / thời tiết) vì không kết nối được cơ sở dữ liệu. "
                "Câu hỏi của bạn: " + user_input
            ),
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    # Ngưỡng ngân sách mặc định nếu truy vấn không nhắc đến giá
    DEFAULT_MAX_PRICE = 5_000_000

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        # TODO 1: Mảng lưu lịch sử traces / tool calls
        self.trace: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def reset_state(self) -> None:
        """Khởi tạo lại lịch sử traces cho mỗi lượt run()."""
        self.trace = []
        self.tool_calls = []

    # ------------------------------------------------------------------
    # TODO 3: Phân tích Thought / Action từ truy vấn tự nhiên -> lập kế hoạch
    # (giả lập suy luận của LLM trước khi đi vào vòng lặp ReAct)
    # ------------------------------------------------------------------
    def build_plan(self, user_input: str) -> List[Dict[str, Any]]:
        lowered = user_input.lower()
        lowered_ascii = strip_accents(lowered)

        # 1) Tìm mã sân bay (HAN/SGN/DAD) xuất hiện trong câu hỏi
        found_codes: List[str] = []
        for code in AIRPORT_CODES:
            if re.search(r"\b" + code + r"\b", user_input, flags=re.IGNORECASE):
                found_codes.append(code)

        # 2) Không có mã -> thử trùng tên thành phố ("Đà Nẵng", "Hà Nội"...)
        if not found_codes:
            for name, code in CITY_NAME_TO_CODE.items():
                if name in lowered or name in lowered_ascii:
                    found_codes.append(code)
                    break

        plan: List[Dict[str, Any]] = []

        # Bước A: Tool tra cứu chuyến bay (nếu truy vấn liên quan chuyến bay/vé)
        needs_flight = bool(
            FLIGHT_KEYWORDS.search(lowered) or FLIGHT_KEYWORDS.search(lowered_ascii)
            or FLIGHT_KEYWORDS.search(strip_accents(user_input))
        )
        if needs_flight and len(found_codes) >= 2:
            plan.append({
                "type": "tool",
                "tool": "get_flight_info",
                "args": {
                    "origin": found_codes[0],
                    "destination": found_codes[1],
                    "max_price": self._parse_max_price(user_input),
                },
            })

        # Bước B: Tool tra cứu thời tiết (nếu truy vấn liên quan thời tiết/trang phục)
        if WEATHER_KEYWORDS.search(lowered) and found_codes:
            plan.append({
                "type": "tool",
                "tool": "get_weather_forecast",
                "args": {"city_code": found_codes[-1]},
            })

        # Fallback: FAQ / ngoài phạm vi -> không cần tool nào
        if not plan:
            plan.append({"type": "faq", "tool": None, "args": {}})

        # Multi-step (nhiều hơn 1 tool) -> cần 1 bước Final Answer riêng
        if len(plan) > 1:
            plan.append({"type": "final", "tool": None, "args": {}})

        return plan

    @staticmethod
    def _parse_max_price(user_input: str) -> int:
        """Phân tích ngân sách kiểu Việt Nam: '2 triệu', '1.5 triệu', '500k'."""
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", user_input, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", "."))
            return int(value * 1_000_000)
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:k|nghìn)\b", user_input, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1_000)
        return ReActAgent.DEFAULT_MAX_PRICE

    # ------------------------------------------------------------------
    # TODO 4: Thực thi Tool trong TOOL_MAP (kèm safeguards chống các bẫy)
    # ------------------------------------------------------------------
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        # Trap 1: Tên tool có thể có khoảng trắng / viết hoa -> chuẩn hoá trước khi tra cứu
        clean_name = (tool_name or "").strip().lower()

        # Trap 2: Tham số Action phải là JSON hợp lệ
        try:
            json.dumps(tool_args)
        except (TypeError, ValueError):
            return {"error": "Invalid JSON format"}

        fn = TOOL_MAP.get(clean_name)
        if fn is None:
            # Báo lỗi Observation thay vì crash -> Agent có thể dừng/đổi hướng
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return fn(**tool_args)
        except Exception as exc:  # Trap 3: Tool lỗi -> trả Observation chứa error
            return {"error": f"Tool execution failed: {exc}"}


    # ------------------------------------------------------------------
    # TODO 2: Vòng lặp ReAct chính (Thought -> Action -> Observation -> ...)
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        self.reset_state()

        # --- Thought ban đầu: phân tích truy vấn và lập kế hoạch các bước ---
        plan = self.build_plan(user_input)

        collected: List[Dict[str, Any]] = []  # Tất cả Observations đã thu được
        iteration = 0
        step_idx = 0
        final_answer: Optional[str] = None

        # --- while iteration < self.max_iterations ---
        while iteration < self.max_iterations:
            iteration += 1
            iteration_num = iteration

            if step_idx >= len(plan):
                # Không còn bước nào trong kế hoạch (trường hợp hiếm)
                break
            step = plan[step_idx]
            step_idx += 1

            if step["type"] == "tool":
                # --- Thought: quyết định bước tiếp theo cần làm gì ---
                thought = (
                    f"Tôi cần gọi tool '{step['tool']}' với tham số "
                    f"{json.dumps(step['args'], ensure_ascii=False)} để lấy dữ liệu cho khách hàng."
                )
                # --- Action: JSON chuẩn {"name": ..., "args": ...} (chống Trap 2) ---
                action = {"name": step["tool"], "args": step["args"]}
                # --- Observation: thực thi tool (chống Trap 1 & 3) ---
                observation = self.execute_tool(step["tool"], step["args"])
                self.tool_calls.append({"tool": step["tool"], "args": step["args"]})
                collected.append(
                    {"tool": step["tool"], "args": step["args"], "observation": observation}
                )

                # Bước kế tiếp vẫn có gì đó (tool khác hoặc bước "final" của
                # multi-step plan)? Nếu đúng thì ghi trace và tiếp tục lặp ReAct.
                next_is_tool = step_idx < len(plan)
                if next_is_tool:
                    self.trace.append({
                        "step": f"iteration_{iteration_num}",
                        "thought": thought,
                        "action": action,
                        "observation": observation,
                    })
                    continue

                # Không còn bước tool nào -> đã có đủ dữ liệu, trả Final Answer luôn
                final_answer = self.compose_final_answer(user_input, collected)
                self.trace.append({
                    "step": f"iteration_{iteration_num}",
                    "thought": thought + " Sau bước truy vấn này tôi đã có đủ dữ liệu.",
                    "action": action,
                    "observation": observation,
                    "final_answer": final_answer,
                })
                break

            elif step["type"] == "final":
                # Bước Final Answer của multi-step plan (sau nhiều tool call)
                final_answer = self.compose_final_answer(user_input, collected)
                self.trace.append({
                    "step": f"iteration_{iteration_num}",
                    "thought": "Tôi đã có đủ dữ liệu từ các tool, tổng hợp câu trả lời cuối cùng.",
                    "action": None,
                    "observation": None,
                    "final_answer": final_answer,
                })
                break

            else:  # type == "faq": không cần tool nào
                final_answer = self.compose_final_answer(user_input, collected)
                self.trace.append({
                    "step": f"iteration_{iteration_num}",
                    "thought": "Câu hỏi này không cần dùng tool nào, tôi trả lời trực tiếp.",
                    "action": None,
                    "observation": None,
                    "final_answer": final_answer,
                })
                break

        # --- Safeguard (Milestone 4): phòng ngừa lặp vô tận (Trap 3) ---
        if final_answer is None:
            return {
                "status": "max_iterations_reached",
                "answer": "Xin lỗi quý khách, tôi không thể hoàn thành yêu cầu trong số bước tối đa cho phép.",
                "iterations": iteration,
                "trace": self.trace,
                "tool_calls": self.tool_calls,
            }

        return {
            "status": "completed",
            "answer": final_answer,
            "iterations": iteration,
            "trace": self.trace,
            "tool_calls": self.tool_calls,
        }

    # ------------------------------------------------------------------
    # TODO 5: Tổng hợp Final Answer từ các Observation đã thu được
    # ------------------------------------------------------------------
    def compose_final_answer(self, user_input: str, collected: List[Dict[str, Any]]) -> str:
        parts: List[str] = []

        flight_obs = next((c for c in collected if c["tool"] == "get_flight_info"), None)
        weather_obs = next((c for c in collected if c["tool"] == "get_weather_forecast"), None)

        if flight_obs is not None:
            flights = flight_obs["observation"]
            if isinstance(flights, list) and flights:
                flight_lines = [
                    f"{fl['flight_number']} ({fl['airline']}) bay lúc {fl['departure_time']} "
                    f"giá {fl['price_vnd']:,} VND"
                    for fl in flights
                ]
                parts.append(
                    "Các chuyến bay phù hợp: " + "; ".join(flight_lines)
                    + f". Tổng cộng tìm thấy {len(flights)} chuyến bay."
                )
            else:
                parts.append(
                    "Rất tiếc hiện không tìm thấy chuyến bay nào phù hợp với yêu cầu của bạn."
                )

        if weather_obs is not None:
            weather = weather_obs["observation"]
            if isinstance(weather, dict) and "error" not in weather:
                parts.append(
                    f"Thời tiết tại {weather['city']}: {weather['temperature_c']}°C, "
                    f"trời {weather['condition'].lower()}, "
                    f"độ ẩm {weather['humidity_pct']}%. "
                    f"Gợi ý trang phục: {weather['recommendation']}"
                )
            else:
                parts.append(
                    "Xin lỗi, tôi chưa lấy được dự báo thời tiết cho điểm đến này lúc này."
                )

        if not parts:
            # FAQ / ngoài phạm vi: không có tool nào được gọi -> trả lời trực tiếp
            return (
                "Xin lỗi, hiện tại tôi chưa có dữ liệu nghiệp vụ riêng về nội dung này "
                f"('{user_input}'). Bạn vui lòng liên hệ tổng đài hỗ trợ hoặc thử "
                "hỏi về chuyến bay / thời tiết nhé."
            )
        return " ".join(parts)

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Answer:", result["answer"])
    print(f'Status: {result["status"]} | Iterations: {result["iterations"]}')
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()