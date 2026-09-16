R=/srv/wizard/runtime/programming-integrated-20260713
C=/srv/wizard/project/target/release/wbrain_compact
echo "=== usage() text does not list --estimate even in current source, so its"
echo "=== absence from the usage banner is not evidence the arm is missing."
echo
for stride in 5000 500 50; do
  echo "--- --estimate stride=$stride ---"
  timeout 900 "$C" --estimate "$R/brain/brain.wbrain" "$stride" 2>&1 | tail -20
  echo "rc=$?"
done
echo "--- --inspect (known blind to slot-table pools) ---"
timeout 300 "$C" --inspect "$R/brain/brain.wbrain" 2>&1 | tail -15
