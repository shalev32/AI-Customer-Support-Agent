"""AI Customer Support Agent — Agentic RAG with OpenAI function calling.

Uses ChromaDB for semantic product search and mongomock for customer data.
Requires OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
"""
import os
import json
import ast
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from customer_store import get_cust_data_collection


# -------------------------------------------------------------------
# Global state & lazy initialization
# -------------------------------------------------------------------
_is_initialized = False
_df = None
_chroma_client = None
_collection = None
_cust_data = None


def _lazy_init():
    global _is_initialized, _df, _chroma_client, _collection, _cust_data
    if _is_initialized:
        return

    # 1. Load customer data
    _cust_data = get_cust_data_collection()

    # 2. Load product data
    product_file = "product_data.csv"
    if not os.path.exists(product_file):
        raise FileNotFoundError(f"{product_file} not found. Please run fetch_product_data.py first.")

    _df = pd.read_csv(product_file)

    # 3. Initialize ChromaDB
    chroma_db_dir = os.path.abspath("chroma_db")
    _chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

    emb_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    _collection = _chroma_client.get_or_create_collection(
        name="products",
        embedding_function=emb_fn
    )

    # If the collection is empty, populate it
    if _collection.count() == 0:
        docs = []
        ids = []
        metadatas = []

        for _, row in _df.iterrows():
            name = str(row.get('Product Name', ''))
            desc = str(row.get('Description', ''))
            sale_price = str(row.get('Sale Price', '0'))
            list_price = str(row.get('List Price', '0'))   # stored for on-sale detection
            brand = str(row.get('Brand', ''))
            category = str(row.get('Category', ''))
            available = str(row.get('Available', ''))

            # Document: "{Product Name}. {Description}"
            doc_text = f"{name}. {desc}"
            docs.append(doc_text)
            ids.append(str(row['Uniq Id']))

            metadatas.append({
                "sale_price": sale_price,
                "list_price": list_price,   # both prices stored → on-sale flagging
                "brand": brand,
                "category": category,
                "available": available,
                "name": name
            })

        _collection.add(documents=docs, ids=ids, metadatas=metadatas)

    _is_initialized = True


# -------------------------------------------------------------------
# Tools implementations
# -------------------------------------------------------------------

def search_products(query: str, k: int = 5) -> str:
    """Semantic (RAG) search over the ChromaDB product vector index."""
    try:
        results = _collection.query(query_texts=[query], n_results=k)

        if not results['documents'] or not results['documents'][0]:
            return "No products found matching the query."

        docs = results['documents'][0]
        metas = results['metadatas'][0]
        ids = results['ids'][0]

        output = []
        for doc, meta, doc_id in zip(docs, metas, ids):
            sale_p = meta.get('sale_price', 'N/A')
            list_p = meta.get('list_price', 'N/A')
            on_sale_flag = ""
            try:
                sp = float(str(sale_p).replace('$', '').replace(',', '').strip())
                lp = float(str(list_p).replace('$', '').replace(',', '').strip())
                if sp < lp:
                    on_sale_flag = " [ON SALE]"
            except Exception:
                pass

            output.append(
                f"- Name: {meta.get('name')}\n"
                f"  Sale Price: ${sale_p}{on_sale_flag}\n"
                f"  List Price: ${list_p}\n"
                f"  Brand: {meta.get('brand')}\n"
                f"  Category: {meta.get('category')}\n"
                f"  Available: {meta.get('available')}\n"
                f"  Description: {doc[:300]}..."
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error in search_products: {str(e)}"


def filter_products(
    max_price: float = None,
    min_price: float = None,
    brand: str = None,
    category: str = None,
    available: str = None,
    name_contains: str = None,
    on_sale: bool = None          # True → Sale Price < List Price
) -> str:
    """Structured filter over product_data.csv for hard attribute constraints."""
    try:
        filtered = _df.copy()

        def safe_float(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, str):
                val = val.replace('$', '').replace(',', '').strip()
            try:
                return float(val)
            except ValueError:
                return 0.0

        if max_price is not None:
            filtered = filtered[filtered['Sale Price'].apply(safe_float) <= max_price]
        if min_price is not None:
            filtered = filtered[filtered['Sale Price'].apply(safe_float) >= min_price]
        if brand is not None:
            filtered = filtered[filtered['Brand'].str.contains(brand, case=False, na=False)]
        if category is not None:
            filtered = filtered[filtered['Category'].str.contains(category, case=False, na=False)]
        if available is not None:
            filtered = filtered[filtered['Available'].astype(str).str.lower() == str(available).lower()]
        if name_contains is not None:
            filtered = filtered[filtered['Product Name'].str.contains(name_contains, case=False, na=False)]
        if on_sale is True:
            filtered = filtered[
                filtered.apply(
                    lambda r: safe_float(r.get('Sale Price')) < safe_float(r.get('List Price')),
                    axis=1
                )
            ]

        if len(filtered) == 0:
            return "No products match the filter criteria."

        head = filtered.head(10)
        output = []
        for _, row in head.iterrows():
            sale_p = row.get('Sale Price', 'N/A')
            list_p = row.get('List Price', 'N/A')
            try:
                on_sale_flag = " [ON SALE]" if safe_float(sale_p) < safe_float(list_p) else ""
            except Exception:
                on_sale_flag = ""

            desc = str(row.get('Description', ''))
            output.append(
                f"- Name: {row.get('Product Name')}\n"
                f"  Sale Price: ${sale_p}{on_sale_flag}\n"
                f"  List Price: ${list_p}\n"
                f"  Brand: {row.get('Brand')}\n"
                f"  Available: {row.get('Available')}\n"
                f"  Category: {row.get('Category')}\n"
                f"  Description: {desc[:200]}"
            )

        res = "\n".join(output)
        if len(filtered) > 10:
            res += f"\n... and {len(filtered) - 10} more."
        return res
    except Exception as e:
        return f"Error in filter_products: {str(e)}"


def get_customer(email: str = None, name: str = None, gift_lookup: bool = False) -> str:
    """
    Look up a customer by email or full name. Never returns the password.

    When gift_lookup=True (looking up a friend's interests for a gift), only
    the non-sensitive fields (name + product-interests) are returned, preventing
    accidental leakage of status, email, or age to a third party.
    """
    try:
        query = {}
        if email:
            query['email'] = email
        elif name:
            parts = name.split()
            if len(parts) >= 2:
                query['first_name'] = {'$regex': f"^{parts[0]}$", '$options': 'i'}
                query['last_name'] = {'$regex': f"^{parts[-1]}$", '$options': 'i'}
            else:
                query['first_name'] = {'$regex': f"^{name}$", '$options': 'i'}

        if not query:
            return "Error: must provide email or name to lookup customer."

        customer = _cust_data.find_one(query)
        if not customer:
            return "Customer not found."

        customer.pop('pw', None)
        customer.pop('_id', None)

        # Gift lookup: return only non-sensitive fields
        if gift_lookup or (name and not email):
            safe_data = {
                "first_name": customer.get("first_name", ""),
                "last_name": customer.get("last_name", ""),
                "product-interests": customer.get("product-interests", [])
            }
            return json.dumps(safe_data, indent=2)

        return json.dumps(customer, indent=2)
    except Exception as e:
        return f"Error in get_customer: {str(e)}"


def authenticate(email: str, pw: str) -> str:
    """Verify a customer's email and password. Returns auth result and status."""
    try:
        customer = _cust_data.find_one({"email": email})
        if not customer:
            return "Authentication failed: Customer not found."
        if customer.get("pw") == pw:
            status = customer.get("status", "regular")
            return f"Authentication successful. Customer status is: {status}."
        else:
            return "Authentication failed: Incorrect password."
    except Exception as e:
        return f"Error in authenticate: {str(e)}"


def read_policies() -> str:
    """Return the text of the store's policies.txt."""
    try:
        if os.path.exists("policies.txt"):
            with open("policies.txt", "r", encoding="utf-8") as f:
                return f.read()
        return "Error: policies.txt not found."
    except Exception as e:
        return f"Error reading policies: {str(e)}"


def calculator(expression: str) -> str:
    """Safe arithmetic evaluation using an AST whitelist walker. No eval()."""
    try:
        class SafeEval(ast.NodeVisitor):
            def visit_BinOp(self, node):
                left = self.visit(node.left)
                right = self.visit(node.right)
                ops = {
                    ast.Add: lambda a, b: a + b,
                    ast.Sub: lambda a, b: a - b,
                    ast.Mult: lambda a, b: a * b,
                    ast.Div: lambda a, b: a / b,
                    ast.Mod: lambda a, b: a % b,
                    ast.Pow: lambda a, b: a ** b,
                    ast.FloorDiv: lambda a, b: a // b,
                }
                op_type = type(node.op)
                if op_type not in ops:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                return ops[op_type](left, right)

            def visit_Num(self, node):          # Python < 3.8 compat
                return node.n

            def visit_Constant(self, node):
                if not isinstance(node.value, (int, float)):
                    raise ValueError(f"Only numeric constants allowed, got: {type(node.value).__name__}")
                return node.value

            def visit_UnaryOp(self, node):
                operand = self.visit(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

            def generic_visit(self, node):
                raise ValueError(f"Unsupported expression: {type(node).__name__}")

        tree = ast.parse(expression, mode='eval')
        result = SafeEval().visit(tree.body)
        return str(result)
    except Exception as e:
        return f"Error in calculator: {str(e)}"


# -------------------------------------------------------------------
# Agent — Tool Schema
# -------------------------------------------------------------------

def get_tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": (
                    "Semantic (RAG) search over the product vector index. "
                    "USE for: vague or natural-language needs, e.g. 'gear to stay safe snowboarding', "
                    "'doctor-recommended soap that won\\'t dry skin', 'something to lose weight without exercise', "
                    "'comparable to brand X'. "
                    "DO NOT USE for: exact product name lookups, specific brand/price/availability filtering — "
                    "use filter_products for those hard constraints instead. "
                    "Always set k large enough (e.g. k=8) when you need to select 2-3 results after filtering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what the customer needs."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return. Default 5; increase to 8-10 when you expect to filter down afterward."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "filter_products",
                "description": (
                    "Structured filter over product_data.csv for hard constraints. "
                    "Price comparisons use Sale Price. "
                    "USE for: exact price limits ('under $100'), specific brand ('brand LHCER'), "
                    "availability filtering, fetching a specific named product, or finding on-sale products. "
                    "DO NOT USE for: vague or natural-language needs — use search_products for those. "
                    "on_sale=true returns only products where Sale Price < List Price."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_price": {"type": "number", "description": "Maximum Sale Price (inclusive)."},
                        "min_price": {"type": "number", "description": "Minimum Sale Price (inclusive)."},
                        "brand": {"type": "string", "description": "Brand name substring (case-insensitive)."},
                        "category": {"type": "string", "description": "Category path substring (case-insensitive)."},
                        "available": {"type": "string", "description": "Availability: 'True' or 'False'."},
                        "name_contains": {"type": "string", "description": "Substring to match in Product Name (case-insensitive)."},
                        "on_sale": {"type": "boolean", "description": "If true, returns only products where Sale Price is strictly below List Price."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_customer",
                "description": (
                    "Look up a customer in the database by email or full name. Never returns the password. "
                    "USE BY EMAIL when the customer asks about their OWN data — but MUST call authenticate() first. "
                    "USE BY NAME with gift_lookup=true when looking up another customer's interests for a gift — "
                    "returns only name and product-interests (non-sensitive). "
                    "DO NOT use to retrieve another customer's status, email, or private data. "
                    "DO NOT skip authentication when the customer asks about their own account data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Customer email address."},
                        "name": {"type": "string", "description": "Customer full name (first and last)."},
                        "gift_lookup": {
                            "type": "boolean",
                            "description": (
                                "Set to true when looking up another customer's interests to buy them a gift. "
                                "Restricts returned fields to name and product-interests only, "
                                "preventing private data (status, email, age) from being revealed."
                            )
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "authenticate",
                "description": (
                    "Verify a customer's identity with their email and password. "
                    "Returns whether authentication succeeded and the customer's premier/regular status. "
                    "ALWAYS call this BEFORE accessing any customer's own private data. "
                    "DO NOT skip this step. "
                    "DO NOT authenticate on behalf of a different customer than the one making the request. "
                    "If authentication fails, do NOT proceed with the private-data request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Customer email address."},
                        "pw": {"type": "string", "description": "Customer password."}
                    },
                    "required": ["email", "pw"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_policies",
                "description": (
                    "Return the full text of the store's company policies. "
                    "USE when uncertain about a store rule (premier threshold, discount rate, no-match policy). "
                    "DO NOT call repeatedly — policies do not change between calls."
                ),
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Safe arithmetic evaluation using a whitelist AST parser. "
                    "ALWAYS use this for ANY arithmetic — never compute in your head. "
                    "Examples: '1700 + 249.99' (premier threshold), '89.99 * 0.95' (5% discount), "
                    "'500 / (4 * 2.5)' (days soap lasts at 4 washes/day x 2.5 mL each). "
                    "Supports: +, -, *, /, //, **, %. "
                    "DO NOT use for string operations or non-numeric expressions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A numeric arithmetic expression, e.g. '11.99 * 0.95'."
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]


# -------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful and accurate customer support AI agent for an online retail store.
You answer free-text customer questions by calling your available tools to search the product catalog,
look up and authenticate customers, and perform arithmetic. You NEVER guess or fabricate.

════════════════════════════════════════════════════════
  MANDATORY COMPANY POLICIES  (follow every single one)
════════════════════════════════════════════════════════

1. BEST MATCH: Always give the customer the best match to their product request from the catalog.

2. PREMIER DISCOUNT: All premier-status customers get exactly 5% off. Compute: price x 0.95.
   Always use the calculator tool for this — never do it in your head.

3. PREMIER THRESHOLD: A customer achieves premier status after $2,000 of purchases in a year.
   To check: calculator("year_to_date_spend + product_sale_price"). If total >= $2,000 -> premier.
   State the math explicitly: current spend, product price, sum, and conclusion.

4. AUTHENTICATION GATE — CRITICAL:
   - If a customer asks about their OWN account data (status, interests, products on sale for them,
     discount, anything from their customer record), MUST call authenticate(email, pw) first.
   - If authentication fails -> refuse immediately, do not reveal any account data.
   - If credentials not provided -> ask for them before proceeding.

5. AUTHORITATIVE CATALOG: All product info comes ONLY from catalog tools (search_products,
   filter_products). NEVER invent, guess, or use your own knowledge for product names, prices,
   or details. Every product you mention MUST have appeared in a tool result in this conversation.

6. MAX 3 SUGGESTIONS: When suggesting products, list AT MOST 3 — TOTAL across ALL categories.
   If the query asks for a specific count ("top 2"), return EXACTLY that count, no more.
   MANDATORY CHECK: Before writing your final answer, count your product suggestions.
   If you have more than 3 (or more than the requested count), REMOVE the extras.
   Never say "let me know if you want me to narrow it down" — narrow it down yourself.

7. PREMIER DISCOUNT + BUDGET CHECK: When an authenticated PREMIER customer has a price limit:
   ALWAYS check if the premier-discounted price fits within their budget BEFORE concluding "no match".
   Step 1: Find the product(s) that best match the request (ignoring price first).
   Step 2: calculator('product_sale_price * 0.95') to get the discounted price.
   Step 3: If discounted price <= budget -> recommend the product at the discounted price.
   Step 4: Only if discounted price STILL exceeds budget -> use the exact no-match line.

- NO-MATCH LINE: When truly no product satisfies the request (including after applying premier
  discount), reply with THIS EXACT SENTENCE and nothing else:
  "We have no product available today that fits your request"
  Do not add anything else. Do not fabricate a product to avoid saying this.

════════════════════════════════════════════════════════
  PRIVACY FIREWALL
════════════════════════════════════════════════════════

A customer's private data (status, password, email, purchase history, discount eligibility)
is NEVER shared with anyone else — not even another authenticated customer.
Each customer can access ONLY their OWN private data.

EXCEPTION: You may look up another customer's product-interests (non-sensitive) to suggest a
gift. Use get_customer(name=..., gift_lookup=True) — returns ONLY name and interests, never
their status, email, or other private fields.

If Customer A asks about Customer B's status/password/private data -> REFUSE.
Say: "I cannot share another customer's account information."

════════════════════════════════════════════════════════
  ON SALE DEFINITION
════════════════════════════════════════════════════════

A product is "on sale" if and only if its Sale Price is STRICTLY less than its List Price.
Tool results mark on-sale products with [ON SALE].

════════════════════════════════════════════════════════
  TOOL SELECTION STRATEGY
════════════════════════════════════════════════════════

- SEMANTIC NEED (vague, natural language, "something like X") -> search_products
- HARD CONSTRAINT (exact price, brand, availability, specific name) -> filter_products
- COMBINE BOTH: e.g. "scales under $100 in lb AND kg" ->
    filter_products(max_price=99.99) + search_products("scales kilograms pounds")

- PRICE SEMANTICS:
    "less than $X" / "under $X" / "cheaper than $X" -> max_price = X - 0.01
    "at most $X" / "up to $X" / "no more than $X"  -> max_price = X

- COMPARABLE PRODUCTS: "comparable to brand X" = same type/use-case (NOT any cheaper product).
    Step 1: filter_products(brand=X) to find anchor product and its price.
    Step 2: search_products("same type as [anchor]") to find similar items.
    Step 3: Keep only truly same kind (e.g. upper-arm BP monitor != wrist monitor).

- INTERESTS-BASED SUGGESTIONS: Exact algorithm when a customer has interest categories:
    Step 1: search_products for the FIRST interest category -> pick the SINGLE best product.
    Step 2: search_products for the SECOND interest category -> pick the SINGLE best product.
    Step 3: search_products for the THIRD interest category -> pick the SINGLE best product.
    Step 4: STOP. Present exactly these 3 products (one per category). Do NOT add more.
    Avoid price outliers (e.g. do NOT pick a $800 item if similar items cost $30-$200).

- ARITHMETIC: ALWAYS use the calculator tool. Never compute in your head.
    For duration questions ("how many days will this last?"), use EXACTLY this formula:
      days = volume_mL / (washes_per_day x mL_per_wash)
      Example: 500 mL soap, 4 washes/day, 2.5 mL/wash -> calculator('500 / (4 * 2.5)') = 50 days.
      To convert oz to mL: 1 oz = 29.57 mL -> calculator('5.5 * 29.57') ~ 162.6 mL.
      Always state: 'Using [expression] = [result]' so the customer can verify.
    Do NOT ask the customer for information you can estimate (~2.5 mL per hand wash default).

- COMPARING PRODUCTS: When comparing specific named products, include for EACH:
    type/kind, key features, price (both Sale and List), and availability.

════════════════════════════════════════════════════════
  PREMIER THRESHOLD — STEP BY STEP
════════════════════════════════════════════════════════

When asked "will buying X make me premier?":
  1. authenticate(email, pw)
  2. filter_products(name_contains="X") to get Sale Price
  3. calculator("year_to_date_spend + product_sale_price")
  4. If result >= 2000 -> "Yes, this purchase will make you a premier customer."
     State: current YTD spend, product price, total, and conclusion.
  5. If result < 2000 -> "No, you need $X more to reach premier status."
"""


# -------------------------------------------------------------------
# Tool Dispatcher
# -------------------------------------------------------------------

def dispatch_tool(tool_call):
    name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Error: Invalid JSON arguments from model."

    dispatch = {
        "search_products": search_products,
        "filter_products": filter_products,
        "get_customer": get_customer,
        "authenticate": authenticate,
        "read_policies": read_policies,
        "calculator": calculator,
    }
    if name not in dispatch:
        return f"Error: Unknown tool '{name}'"
    return dispatch[name](**kwargs)


# -------------------------------------------------------------------
# Main agent entry point
# -------------------------------------------------------------------

def answer_question(question: str) -> str:
    """
    Main agent entry point. Called once per query.
    Import-safe: all heavy initialization deferred to first call.
    """
    try:
        _lazy_init()
    except Exception as e:
        return f"Initialization Error: {str(e)}"

    deployment_name = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    endpoint = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return "Error: OPENAI_API_KEY environment variable is not set. See .env.example."
    if not endpoint:
        return "Error: OPENAI_BASE_URL environment variable is not set. See .env.example."

    try:
        client = OpenAI(api_key=api_key, base_url=endpoint, max_retries=3)
    except Exception as e:
        return f"Error setting up OpenAI client: {str(e)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    tools = get_tools_schema()

    # Hard cap: 15 iterations to prevent infinite loops
    for _ in range(15):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                tools=tools,
                temperature=0.0,
            )
        except Exception as e:
            err_str = str(e)
            if "BadRequestError" in str(type(e)) or "content_filter" in err_str.lower():
                messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM: The previous request was blocked by the content filter. "
                        f"Please rephrase your approach. Error: {err_str}"
                    )
                })
                continue
            return f"API Error: {err_str}"

        choice = response.choices[0]

        # Final answer: text with no tool calls
        if choice.message.content and not choice.message.tool_calls:
            return choice.message.content

        messages.append(choice.message)

        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                tool_response = dispatch_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(tool_response)
                })
        elif not choice.message.content:
            # Azure occasionally returns empty message — nudge for final answer
            messages.append({
                "role": "user",
                "content": "Please provide your final answer to the customer based on the tool results above."
            })

    # Exceeded cap — return last assistant text if available
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
            return msg.content
        if isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('content'):
            return msg['content']

    return "Error: Reached maximum tool call limit without producing a final answer."
