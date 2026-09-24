from __future__ import annotations
import argparse,json
from sizon.data.data import DataFeed
from sizon.research.evolution import Engine

def main():
    parser=argparse.ArgumentParser(prog="sizon",description="Reproducible evolutionary alpha research")
    sub=parser.add_subparsers(dest="command",required=True)
    p_validate=sub.add_parser("validate"); p_validate.add_argument("path")
    p_run=sub.add_parser("run"); p_run.add_argument("path"); p_run.add_argument("--output-dir",default="runs"); p_run.add_argument("--run-id"); p_run.add_argument("--population",type=int,default=20); p_run.add_argument("--generations",type=int,default=5); p_run.add_argument("--seed",type=int,default=7)
    args=parser.parse_args(); feed=DataFeed.from_path(args.path)
    if args.command=="validate": print(json.dumps(feed.validate(),indent=2)); return
    engine=Engine(feed,population=args.population,generations=args.generations,seed=args.seed,output_dir=args.output_dir,run_id=args.run_id); best=engine.run()
    print(json.dumps({"run_id":engine.store.run_id,"run_path":str(engine.store.path),"best_train_metrics":best.scores,"strategies_saved":args.population*args.generations},indent=2))
if __name__=="__main__": main()
