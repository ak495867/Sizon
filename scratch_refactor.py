import re

with open(r'd:\Sizon\sizon\core\expression.py', 'r') as f:
    content = f.read()

start_str = '        if name == "EMA":'
end_str = '        # Registered experimental primitives'
start_idx = content.find(start_str)
end_idx = content.find(end_str)

evaluate_body = content[start_idx:end_idx]

# match "if name == 'something':" or "if name in {'a', 'b'}:" 
# block goes until the next "if name " or the end of the evaluate_body
pattern = re.compile(r'if name (==|in) (.*?):\n(.*?)(?=\n        if name (?:==|in) |\Z)', re.DOTALL)
matches = pattern.findall(evaluate_body)

kernels = []
registry_entries = []

for op, cond, body in matches:
    names = list(eval(cond.strip())) if op == 'in' else [eval(cond.strip())]
    func_name = "_kernel_" + names[0].lower().replace(" ", "_")
    
    # ensure proper indentation
    lines = body.split('\n')
    kernel_code = f"def {func_name}(p, close, high, low, open_, volume, ctx):\n"
    for line in lines:
        if line.strip():
            # strip 12 spaces (the body is indented by 12 spaces originally)
            if line.startswith("            "):
                kernel_code += "    " + line[12:] + "\n"
            else:
                kernel_code += "    " + line.strip() + "\n"
        else:
            kernel_code += "\n"
            
    kernels.append(kernel_code)
    for n in names:
        registry_entries.append(f'    "{n}": {func_name},')

kernels_str = "\n".join(kernels)
registry_str = "KERNELS = {\n" + "\n".join(registry_entries) + "\n}\n"

new_evaluate = '''
    def evaluate(self, ctx):
        name = self.name.upper()
        p = max(2, int(self.period))
        close = ctx.columns["close"]
        high = ctx.columns.get("high", close)
        low = ctx.columns.get("low", close)
        open_ = ctx.columns.get("open", close)
        volume = ctx.columns.get("volume", [1.0] * len(close))

        if name in KERNELS:
            return KERNELS[name](p, close, high, low, open_, volume, ctx)
'''

prim_idx = content.find('@dataclass(frozen=True)\nclass Primitive(Node):')
final_eval_end = content.find('        raise ValueError(f"Unknown primitive: {self.name}")')

new_content = content[:prim_idx] + kernels_str + "\n" + registry_str + "\n" + content[prim_idx:start_idx-1] + new_evaluate + content[end_idx:final_eval_end] + '        raise ValueError(f"Unknown primitive: {self.name}")\n' + content[final_eval_end + len('        raise ValueError(f"Unknown primitive: {self.name}")'):]

with open(r'd:\Sizon\sizon\core\expression.py', 'w') as f:
    f.write(new_content)
