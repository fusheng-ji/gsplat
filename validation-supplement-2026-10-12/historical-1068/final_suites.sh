set -u
source /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/env.sh
cd /data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat/.claude/worktrees/gsplat-open-prs-bugs-930cbc
export PYTHONPATH=$(pwd):$PYTHONPATH
for ref in main 61003a2 8b329df 1880e3e 0f91ca6; do
  git checkout -q --detach $ref
  python -c "import gsplat.cuda._backend" >/dev/null 2>&1
  python -m pytest -q -rfE -p no:cacheprovider tests/test_basic.py tests/test_rasterization.py tests/test_2dgs.py tests/test_2dgs_scene_integration.py > /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/final-suite-$ref.log 2>&1
  grep "^FAILED" /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/final-suite-$ref.log | sed 's/ - .*//' | sort > /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/final-failed-$ref.txt
  echo "$ref: $(tail -1 /tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad/final-suite-$ref.log)"
done
git checkout -q fix/2dgs-median-rgb-mode
echo SUITES_DONE
