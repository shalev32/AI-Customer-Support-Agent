"""AI Customer Support Agent — Agentic RAG with OpenAI function calling.

Uses ChromaDB for semantic product search and mongomock for customer data.
Requires OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
"""
import os
import json
import ast
import traceback
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
            # Document representation: "{Product Name}. {Description}"
            # Ensure no NaNs
            name = str(row.get('Product Name', ''))
            desc = str(row.get('Description', ''))
            price = str(row.get('Sale Price', '0'))
            brand = str(row.get('Brand', ''))
            category = str(row.get('Category', ''))
            available = str(row.get('Available', ''))
            
            doc_text = f"{name}. {desc}"
            docs.append(doc_text)
            ids.append(str(row['Uniq Id']))
            
            metadatas.append({
                "price": price,
                "brand": brand,
                "category": category,
                "available": available,
                "name": name
            })
            
        # Add to collection in batches (Chroma handles batching internally, but we can just pass it all)
        _collection.add(
            documents=docs,
            ids=ids,
            metadatas=metadatas
        )

    _is_initialized = True

# -------------------------------------------------------------------
# Tools implementations
# -------------------------------------------------------------------

def search_products(query: str, k: int = 5) -> str:
    try:
        results = _collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No products found matching the query."
            
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        ids = results['ids'][0]
        
        output = []
        for doc, meta, doc_id in zip(docs, metas, ids):
            output.append(
                f"- Name: {meta.get('name')}\n"
                f"  Price: ${meta.get('price')}\n"
                f"  Brand: {meta.get('brand')}\n"
                f"  Category: {meta.get('category')}\n"
                f"  Available: {meta.get('available')}\n"
                f"  Description: {doc[:200]}..."
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error in search_products: {str(e)}"

def filter_products(max_price: float = None, min_price: float = None, 
                   brand: str = None, category: str = None, 
                   available: str = None, name_contains: str = None) -> str:
    try:
        # Work on a copy of the dataframe
        filtered = _df.copy()
        
        # Helper to convert price to float safely
        def safe_float(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, str):
                val = val.replace('$', '').replace(',', '').strip()
            try:
                return float(val)
            except ValueError:
                return 0.0

        # Apply hard constraints
        if max_price is not None:
            filtered = filtered[filtered['Sale Price'].apply(safe_float) <= max_price]
        if min_price is not None:
            filtered = filtered[filtered['Sale Price'].apply(safe_float) >= min_price]
            
        if brand is not None:
            # Case-insensitive brand match
            filtered = filtered[filtered['Brand'].str.contains(brand, case=False, na=False)]
            
        if category is not None:
            filtered = filtered[filtered['Category'].str.contains(category, case=False, na=False)]
            
        if available is not None:
            filtered = filtered[filtered['Available'].astype(str).str.lower() == str(available).lower()]
            
        if name_contains is not None:
            filtered = filtered[filtered['Product Name'].str.contains(name_contains, case=False, na=False)]
            
        if len(filtered) == 0:
            return "No products match the filter criteria."
            
        # Return top 10 matches max
        head = filtered.head(10)
        output = []
        for _, row in head.iterrows():
            output.append(
                f"- Name: {row.get('Product Name')}\n"
                f"  Price: ${row.get('Sale Price')}\n"
                f"  Brand: {row.get('Brand')}\n"
                f"  Available: {row.get('Available')}\n"
                f"  Category: {row.get('Category')}"
            )
            
        res = "\n".join(output)
        if len(filtered) > 10:
            res += f"\n... and {len(filtered) - 10} more."
        return res
    except Exception as e:
        return f"Error in filter_products: {str(e)}"

def get_customer(email: str = None, name: str = None) -> str:
    try:
        query = {}
        if email:
            query['email'] = email
        elif name:
            # Splitting full name into first and last for simple matching
            parts = name.split()
            if len(parts) >= 2:
                # Case-insensitive regex
                query['first_name'] = {'$regex': f"^{parts[0]}$", '$options': 'i'}
                query['last_name'] = {'$regex': f"^{parts[-1]}$", '$options': 'i'}
            else:
                query['first_name'] = {'$regex': f"^{name}$", '$options': 'i'}
                
        if not query:
            return "Error: must provide email or name to lookup customer."
            
        customer = _cust_data.find_one(query)
        if not customer:
            return "Customer not found."
            
        # Never return the password
        if 'pw' in customer:
            del customer['pw']
        if '_id' in customer:
            del customer['_id']
            
        return json.dumps(customer, indent=2)
    except Exception as e:
        return f"Error in get_customer: {str(e)}"

def authenticate(email: str, pw: str) -> str:
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
    try:
        if os.path.exists("policies.txt"):
            with open("policies.txt", "r", encoding="utf-8") as f:
                return f.read()
        return "Error: policies.txt not found."
    except Exception as e:
        return f"Error reading policies: {str(e)}"

def calculator(expression: str) -> str:
    try:
        # whitelist AST walker
        class SafeEval(ast.NodeVisitor):
            def visit_BinOp(self, node):
                left = self.visit(node.left)
                right = self.visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                else:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

            def visit_Num(self, node):
                return node.n

            def visit_Constant(self, node):
                return node.value

            def visit_UnaryOp(self, node):
                operand = self.visit(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.USub):
                    return -operand
                else:
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
                    
            def generic_visit(self, node):
                raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

        tree = ast.parse(expression, mode='eval')
        evaluator = SafeEval()
        result = evaluator.visit(tree.body)
        return str(result)
    except Exception as e:
        return f"Error in calculator: {str(e)}"

# -------------------------------------------------------------------
# Agent Logic
# -------------------------------------------------------------------

def get_tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "RAG semantic search over the product vector index. Returns the top-k products. Use for natural-language needs (e.g. 'lose weight without exercise', 'soap that won\\'t dry skin', 'something like X').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query to search for."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return. Default is 5."
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
                "description": "Structured filter over product_data.csv for hard constraints. Price uses Sale Price. Use for queries like 'under $100', 'brand LHCER', or to fetch a specific named product. Do not use for semantic queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_price": {
                            "type": "number",
                            "description": "Maximum sale price"
                        },
                        "min_price": {
                            "type": "number",
                            "description": "Minimum sale price"
                        },
                        "brand": {
                            "type": "string",
                            "description": "Brand name"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category name"
                        },
                        "available": {
                            "type": "string",
                            "description": "Availability status (e.g., 'True' or 'False')"
                        },
                        "name_contains": {
                            "type": "string",
                            "description": "Substring to search in product name"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_customer",
                "description": "Look up a customer by email or full name. Never returns the password. Note: to answer questions about a customer's own private data (status, discount), you must call authenticate() first. But to look up another person's product-interests (to buy a gift for them), no authentication is needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string"
                        },
                        "name": {
                            "type": "string"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "authenticate",
                "description": "Verify a customer's identity. Returns whether authenticated and the customer's status (premier or regular). You MUST call this before revealing the customer's own account data or applying their discount.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string"
                        },
                        "pw": {
                            "type": "string"
                        }
                    },
                    "required": ["email", "pw"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_policies",
                "description": "Return the text of the company policies. Call this if you are unsure about the store's rules.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Safe arithmetic evaluation. Use this to do ANY math. Never do math in your head! Examples: '1700 + 49.99' (to check premier threshold), '49.99 * 0.95' (for 5% discount), '120 / 4' (for days it will last). Supports +, -, *, /, **, %.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate, e.g. '11.99 * 0.95'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

SYSTEM_PROMPT = """You are a highly capable customer support AI agent for an online store.
Your job is to answer customer questions accurately by using your available tools to search the product catalog, lookup customers, authenticate, and do math.

IMPORTANT: You MUST ALWAYS follow these 7 company policies as actionable rules:
1. A customer seeking a product should always be given the best match to their request.
2. All customers with premier status get a 5% discount on items. (Compute: Price * 0.95)
3. A customer gets premier status after $2000 of purchases in a year. The status lasts one year. (To check if a new purchase makes them premier, add their stated year-to-date spend to the item's price and compare to $2000).
4. If a customer asks for information regarding their own customer data (their status, discount, what is on sale for them), they must provide their email and pw. You MUST authenticate them first. If authentication fails, do not reveal the data. (Looking up another person's product-interests to buy them a gift is allowed without auth).
5. Product information from the catalog is authoritative. Never fabricate products, prices, or info from the internet.
6. If a customer asks you to suggest a product, provide a list of NO MORE THAN 3 products that best fit the request. (If they ask for top 2, give exactly 2).
7. If no product satisfies the request, tell them EXACTLY: "We have no product available today that fits your request". Do not add anything else to this sentence.

STRATEGY GUIDANCE:
- For natural language needs (e.g. "soap that doesn't dry skin", "gear to stay safe snowboarding"), use search_products (semantic RAG).
- For hard constraints ("under $100", "brand X", specific named products), use filter_products.
- You can combine them! E.g. filter_products for "under $100", then search_products for semantic matching.
- ALWAYS use the calculator tool for math. Never do math in your head!
- ALWAYS ground your answers in the data returned by the tools.

THINKING: Before you call a tool, you may optionally reason step-by-step by returning a short thinking block before your tool call.
When you have the final answer, just return it plainly to the user.
"""

def dispatch_tool(tool_call):
    name = tool_call.function.name
    try:
        kwargs = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Error: Invalid JSON arguments."
        
    if name == "search_products":
        return search_products(**kwargs)
    elif name == "filter_products":
        return filter_products(**kwargs)
    elif name == "get_customer":
        return get_customer(**kwargs)
    elif name == "authenticate":
        return authenticate(**kwargs)
    elif name == "read_policies":
        return read_policies(**kwargs)
    elif name == "calculator":
        return calculator(**kwargs)
    else:
        return f"Error: Unknown tool {name}"

def answer_question(question: str) -> str:
    # Ensure lazy initialization is done
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
        client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
            max_retries=3
        )
    except Exception as e:
        return f"Error setting up OpenAI client: {str(e)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    tools = get_tools_schema()
    
    # 15 iterations maximum to prevent infinite loops
    for _ in range(15):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                tools=tools,
                temperature=0.0,
            )
        except Exception as e:
            # Handle BadRequestError (content filter) or other errors
            if "BadRequestError" in str(type(e)):
                # Let the agent see the error so it can rephrase
                messages.append({"role": "user", "content": f"SYSTEM ERROR: The previous message triggered the Azure content filter. Please rephrase without sensitive words. Error details: {str(e)}"})
                continue
            else:
                return f"API Error: {str(e)}"
                
        choice = response.choices[0]
        
        # If the model produced a normal text response (without or alongside tool calls)
        if choice.message.content:
            # If there are no tool calls, this is the final answer!
            if not choice.message.tool_calls:
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
        else:
            # Fallback if no tool calls and no content (shouldn't happen, but just in case)
            if not choice.message.content:
                return "Error: Unexpected empty response from model."
                
    # If we hit the 15 call cap
    return "Error: Reached maximum tool call limit without resolving the answer."

