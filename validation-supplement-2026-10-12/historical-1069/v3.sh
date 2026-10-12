set -u
S=/tmp/claude-9025/-data-wrd-projects-students-wenbo-ji-egocentric-video-generation-gsplat--claude-worktrees-gsplat-open-prs-bugs-930cbc/609bda1f-3faa-4ea0-95a9-15a7ecb329d4/scratchpad
W=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/gsplat/.claude/worktrees/gsplat-open-prs-bugs-930cbc
O=$S/v3; cd $W
source $S/env.sh; export PYTHONPATH=$W:$PYTHONPATH MAX_JOBS=48
T="python -m pytest -p no:cacheprovider"
build() { python -c "import gsplat.cuda._backend" > $O/build-$1.log 2>&1; echo "build at $1: $(grep -c 'set up successfully' $O/build-$1.log) rebuilds, exit=$?"; }
git checkout -q --detach origin/fix/2dgs-multicam-nonpacked; build 3acaba8
# 1. formatter on the new heads
for ref in 78d4c42 3acaba8; do git checkout -q --detach $ref; ( export PATH=$S/fmt/bin:$PATH PYTHONPATH=$S/fmt; bash lint/format-code.sh --check --full > $O/format-$ref.log 2>&1; echo "format $ref exit=$?" ); done
# 2a. multicam test on its branch
git checkout -q --detach 3acaba8; build 3acaba8
echo "multicam @3acaba8: $($T -q -rA tests/test_2dgs.py -k multi_camera 2>&1 | grep -E '^(PASSED|FAILED|XFAIL|XPASS|ERROR)' | sed 's/ - .*//' | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')"
$T -q -rA tests/test_2dgs.py -k multi_camera > $O/mc-3acaba8.log 2>&1; grep -E "^(PASSED|FAILED|XFAIL|XPASS)" $O/mc-3acaba8.log | sed 's/ - .*//'; grep -E "^XFAIL" $O/mc-3acaba8.log | head -2
# 2b. multicam test on main
git checkout -q --detach main; build main; git show 3acaba8:tests/test_2dgs.py > tests/test_zz_mc.py
$T -q -rA tests/test_zz_mc.py -k multi_camera > $O/mc-main.log 2>&1; rm -f tests/test_zz_mc.py
echo "multicam test on main: $(tail -1 $O/mc-main.log) | errors: $(grep -E '^E +RuntimeError' $O/mc-main.log | sort | uniq -c | tr '\n' ' ')"
# 2c. lidar test on its branch and on main
git checkout -q --detach 78d4c42; build 78d4c42
echo "lidar @78d4c42: $($T -q tests/test_rasterization.py -k lidar_tiling_fits 2>&1 | tail -1)"
git checkout -q --detach main; build main; git show 78d4c42:tests/test_rasterization.py > tests/test_zz_lidar.py
$T -q tests/test_zz_lidar.py -k lidar_tiling_fits > $O/lidar-main.log 2>&1; rm -f tests/test_zz_lidar.py
echo "lidar test on main: $(tail -1 $O/lidar-main.log)"; grep -E "^E  |test_zz_lidar.py:[0-9]+" $O/lidar-main.log | head -8
# 2d. merges
git checkout -q --detach 8b329df; git merge -q --no-edit 3acaba8 -m tmp >/dev/null; build merge-1068-1070
echo "multicam with #1068 merged: $($T -q -rA tests/test_2dgs.py -k multi_camera 2>&1 | grep -E '^(PASSED|FAILED|XFAIL|XPASS)' | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')"
git checkout -q --detach 8b329df; git merge -q --no-edit 78d4c42 -m tmp1 >/dev/null; git merge -q --no-edit 3acaba8 -m tmp2 >/dev/null; build merge-all
$T -q -rA tests/test_2dgs.py tests/test_rasterization.py -k "depth_backward or median_depth_rgb_mode or compute_median or multi_camera or lidar_tiling_fits" > $O/targeted-all.log 2>&1
echo "all three merged, targeted: $(tail -1 $O/targeted-all.log) | outcomes: $(grep -E '^(PASSED|FAILED|XFAIL|XPASS|ERROR)' $O/targeted-all.log | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')"
# 3. full suites on new heads
for ref in 78d4c42 3acaba8; do
  git checkout -q --detach $ref; build $ref
  ( export CUDA_VISIBLE_DEVICES=""; timeout 10800 $T -q -rfE tests/ > $O/cpu-$ref.log 2>&1; echo "cpu $ref: $(tail -1 $O/cpu-$ref.log)" )
  timeout 21600 $T -q -rfE tests/ > $O/gpu-$ref.log 2>&1; echo "gpu $ref: $(tail -1 $O/gpu-$ref.log)"
  for k in cpu gpu; do grep -E "^(FAILED|ERROR)" $O/$k-$ref.log | sed 's/ - .*//' | sort > $O/$k-failed-$ref.txt; done
done
git checkout -q fix/2dgs-median-rgb-mode
echo V3_DONE
