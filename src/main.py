import os
import argparse

from logicalbeam.vanilla import save_t2s_outputs_many
from logicalbeam.template_gen import save_templates_beamsearch, save_templates_logicalbeam
from logicalbeam.template_fill import save_filled_in_templates
from logicalbeam.ablation_direct import save_direct_t2s_with_plan_without_branching_control, \
    save_direct_t2s_with_plan_with_branching_control

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", help="One of text-to-sql-vanilla, " + \
        "text-to-template, fill-in-template and text-to-sql-plan", default="text-to-template")
    parser.add_argument("--method", "-t", help="beam-search or logicalbeam, for " + \
        "template generation or infilling", default="logicalbeam")
    parser.add_argument("--branch-control", "-b", help="Use branching control, " \
        + "if using the direct text-to-sql methods with plans", action='store_true')
    parser.add_argument("--split-or-path", "-s", help="The dataset split or input path", \
        default="validation")
    parser.add_argument("--checkpt-path", "-c", help="The path to the checkpoint to use", \
        default=None)
    parser.add_argument("--out-path", "-o", help="The file where the outputs will be saved", \
        default=None)
    parser.add_argument("--beam-width", "-w", help="Beam width", \
        type=int, default=None)
    parser.add_argument("--num-outputs", "-n", help="#outputs to save per instance", \
        type=int, default=None)
    parser.add_argument("--column", "-bc", help="Allow branching at columns " + \
        "(fill-in w/ logicalbeam)", action='store_true')
    parser.add_argument("--table", "-bt", help="Allow branching at table " + \
        "(fill-in w/ logicalbeam)", action='store_true')
    
    args = parser.parse_args()
    assert args.mode in ["text-to-sql-vanilla", "text-to-template", "fill-in-template", \
        "text-to-sql-plan"], "Invalid mode!"
    assert args.method in ["beam-search", "logicalbeam"], "Invalid method for generation/infilling!"
    if args.mode == "text-to-sql-vanilla":
        if args.checkpt_path is not None:
            print("WARNING: A checkpoint path has been provided for vanilla text-to-sql.")
            print("Note that our flan-t2s checkpoint is *not* the vanilla model.")
            print("We use the PICARD checkpoint from the HuggingFace hub as the vanilla model.")
    else:
        assert args.checkpt_path is not None, "No checkpoint specified!"
    
    # Defaults
    if args.beam_width is None:
        if args.mode == "text-to-template" and args.method == "logicalbeam":
            # lightweight
            args.beam_width = 1
        else:
            args.beam_width = 10
    
    if args.num_outputs is None:
        if args.mode == "fill-in-template":
            # num_outputs is used pre-filtration; no point in throwing stuff at that stage
            # evaluation is always w.r.t top-5.
            args.num_outputs = args.beam_width
        else:
            args.num_outputs = 5
    
    return args

def main():
    args = parse_args()
    if args.mode == "text-to-sql-vanilla":
        save_t2s_outputs_many(
            split=args.split_or_path,
            with_content=False,
            checkpt_path=args.checkpt_path,
            out_path=args.out_path,
            beam_width=args.beam_width,
            num_outputs=args.num_outputs
        )
    elif args.mode == "text-to-template":
        if args.method == "beam-search":
            save_templates_beamsearch(
                split=args.split_or_path,
                with_content=False,
                checkpt_path=args.checkpt_path,
                out_path=args.out_path,
                beam_width=args.beam_width,
                num_outputs=args.num_outputs
            )
        else:
            save_templates_logicalbeam(
                split=args.split_or_path,
                with_content=False,
                checkpt_path=args.checkpt_path,
                out_path=args.out_path,
                beam_width=args.beam_width
            )
    elif args.mode == "fill-in-template":
        controlled = (args.method == "logicalbeam")
        save_filled_in_templates(
            in_path=args.split_or_path,
            with_content=False,
            checkpt_path=args.checkpt_path,
            out_path=args.out_path,
            beam_width=args.beam_width,
            num_outputs=args.num_outputs,
            controlled=controlled,
            column=args.column,
            table=args.table
        )
    else:
        if args.branch_control:
            save_direct_t2s_with_plan_with_branching_control(
                split_or_path=args.split_or_path,
                out_path=args.out_path,
                beam_width=args.beam_width,
                column=args.column,
                table=args.table,
                checkpt_path=args.checkpt_path
            )
        else:
            save_direct_t2s_with_plan_without_branching_control(
                split_or_path=args.split_or_path,
                out_path=args.out_path,
                beam_width=args.beam_width,
                checkpt_path=args.checkpt_path
            )
        
    
if __name__ == '__main__':
    main()