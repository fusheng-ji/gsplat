set -u
source /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/env.sh
W=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat/.claude/worktrees/gsplat-open-prs-bugs-930cbc
cd $W
D=/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/data/360_v2/garden; CK=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/results/tip/garden/ckpts/ckpt_29999.pt; O=/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/realcase; mkdir -p $O
PY=$(which python)
# 1. trainer, exact command used before
for ref in main 8b329df; do
  git checkout -q --detach $ref; $PY -c "import gsplat.cuda._backend" >/dev/null 2>&1
  (cd examples && PYTHONPATH=/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/trainer-libs:$W:$PYTHONPATH timeout 1800 $PY simple_trainer_2dgs.py --disable_viewer --data_dir $D --data_factor 4 --result_dir $O/train-$ref --max_steps 200 --eval_steps 100000 --save_steps 100000 > $O/trainer-$ref.log 2>&1; echo "trainer $ref exit=$?")
  tr '\r' '\n' < $O/trainer-$ref.log | grep -E "loss=|RuntimeError" | sed -n '1p;$p' | cut -c1-120
done
# 2. renders from the kept garden checkpoint
probe() { ref=$1; shift; git checkout -q --detach $ref; $PY -c "import gsplat.cuda._backend" >/dev/null 2>&1; PYTHONPATH=$W:$W/examples:/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/trainer-libs:$PYTHONPATH $PY /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/render_probe.py --ckpt $CK --data_dir $D --factor 4 "$@" 2>&1 | grep -E "^saved|Error" | sed "s|/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/||"; }
for ref in main 61003a2 8b329df; do probe $ref --mode median --views 0 --out $O/median_$ref.npz; done
for ref in main 0f91ca6; do probe $ref --mode multicam --views 0,1,2 --out $O/multicam_$ref.npz; done
git checkout -q fix/2dgs-median-rgb-mode
for fix in 61003a2 8b329df; do echo "median main vs $fix: $($PY /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/make_figs.py --kind median --main $O/median_main.npz --fix $O/median_$fix.npz --view 0 --title x --out $O/fig_median_$fix.png)"; done
echo "multicam main vs 0f91ca6: $($PY /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/make_figs.py --kind multicam --main $O/multicam_main.npz --fix $O/multicam_0f91ca6.npz --views 0,1,2 --title x --out $O/fig_multicam.png)"
echo "previous figure stats:"; cat /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/figs/pr1_issue714_median_garden.json; echo; cat /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat-pr-validation/figs/pr3_multicam_garden.json; echo
echo REALCASE_DONE
