#!/usr/bin/env bash
# Bash script to run all permutations for breakthrough game permutations.

stat out 1>/dev/null 2>&1
if [ $? != 0 ]; then
    echo "Creating directory 'out'"
    mkdir ./out
fi

echo "Playing all games for report."
echo "Output will be placed into txt files in the 'out' directory."
echo "-----------"
echo ""

# Minimax v. minimax
echo "Playing minimax v. minimax games..."
echo "    minimax (offensive) v. minimax (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 minimax -e1 offensive -e2 defensive > out/mmOffvmmDef.txt

echo "    minimax (defensive) v. minimax (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 minimax -e1 defensive -e2 offensive > out/mmDefvmmOff.txt

echo "    minimax (offensive) v. minimax (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 minimax -e1 offensive -e2 offensive > out/mmOffvmmOff.txt

echo "    minimax (defensive) v. minimax (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 minimax -e1 defensive -e2 defensive > out/mmDefvmmDef.txt

# Alpha-beta v. alpha-beta
echo "Playing alphabeta v. alphabeta games..."
echo "    alphabeta (offensive) v. alphabeta (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 alphabeta -e1 offensive -e2 defensive > out/abOffvabDef.txt

echo "    alphabeta (defensive) v. alphabeta (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 alphabeta -e1 defensive -e2 offensive > out/abDefvabOff.txt

echo "    alphabeta (offensive) v. alphabeta (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 alphabeta -e1 offensive -e2 offensive > out/abOffvabOff.txt

echo "    alphabeta (defensive) v. alphabeta (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 alphabeta -e1 defensive -e2 defensive > out/abDefvabDef.txt

# Minimax v. alpha-beta
echo "Playing minimax v. alphabeta games..."
echo "    minimax (offensive) v. alphabeta (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 alphabeta -e1 offensive -e2 defensive > out/mmOffvabDef.txt

echo "    minimax (defensive) v. alphabeta (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 alphabeta -e1 defensive -e2 offensive > out/mmDefvabOff.txt

echo "    minimax (offensive) v. alphabeta (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 alphabeta -e1 offensive -e2 offensive > out/mmOffvabOff.txt

echo "    minimax (defensive) v. alphabeta (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 minimax -s2 alphabeta -e1 defensive -e2 defensive > out/mmDefvabDef.txt

# Alpha-beta v. minimax
echo "Playing alphabeta v. minimax games..."
echo "    alphabeta (offensive) v. minimax (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 minimax -e1 offensive -e2 defensive > out/abOffvmmDef.txt

echo "    alphabeta (defensive) v. minimax (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 minimax -e1 defensive -e2 offensive > out/abDefvmmOff.txt

echo "    alphabeta (offensive) v. minimax (offensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 minimax -e1 offensive -e2 offensive > out/abOffvmmOff.txt

echo "    alphabeta (defensive) v. minimax (defensive)"
python breakthrough.py -d1 3 -d2 3 -s1 alphabeta -s2 minimax -e1 defensive -e2 defensive > out/abDefvmmDef.txt