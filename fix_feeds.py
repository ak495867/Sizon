import os
import re

def fix_feeds():
    pattern = re.compile(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):(\s+)feed\s*=\s*DataFeed\.from_csv\([\"']examples/sample\.csv[\"']\)")
    
    def replacer(match):
        func_name = match.group(1)
        args = match.group(2)
        space = match.group(3)
        
        if "sample_feed" not in args:
            if args.strip() == "":
                new_args = "sample_feed"
            else:
                new_args = args + ", sample_feed"
        else:
            new_args = args
            
        return f"def {func_name}({new_args}):{space}feed = sample_feed"

    for root, _, files in os.walk("d:/Sizon/tests"):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                
                new_content = pattern.sub(replacer, content)
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(new_content)
                    print(f"Updated {path}")

if __name__ == "__main__":
    fix_feeds()
