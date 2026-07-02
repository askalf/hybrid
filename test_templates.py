#!/usr/bin/env python3
"""templates.py tests — the five shapes answer exactly; everything else DECLINES.

Like every oracle test in this repo: pure Python, no model, no network. The pass
cases are the live bench/holdout/stress queries the model tiers used to spend
5-40 s on; the decline cases are the traps that MUST fall through to the model
tiers (a template that guesses on a riddle is worse than no template at all)."""
import sys
from templates import solve

PASS = [
    # rate — per-item and per-time products (the confident-wrong-multiplication class)
    ("Each pallet holds 3,672 cans. How many cans are on 38 pallets?", "139536", "rate"),
    ("Each shipping container holds 1,728 units. How many units in 56 containers?", "96768", "rate"),
    ("Each fuel tank holds 13.9 liters. How many liters do 73 tanks hold?", "1014.7", "rate"),
    ("Each crate weighs 23.7 kg. What do 41 crates weigh?", "971.7", "rate"),
    ("A printer prints 2,417 pages per hour. How many pages in 94 hours?", "227198", "rate"),
    ("A factory makes 1,847 widgets per day. How many widgets in 263 days?", "485761", "rate"),
    ("A server processes 3,408 requests per minute. How many in 47 minutes?", "160176", "rate"),
    ("A warehouse ships 2,963 boxes per day. How many boxes does it ship in 187 days?", "554081", "rate"),
    ("A data center draws 1,384 watts per rack. What is the total draw of 219 racks?", "303096", "rate"),
    ("A book costs $18.75. How much do 4 books cost?", "75", "rate"),
    ("A store sells notebooks at $12.50 each. How much do 7 notebooks cost?", "87.5", "rate"),
    # sum-diff — the bat-and-ball family, never the $0.10 trap answer
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?", "0.05", "sum-diff"),
    ("A bat and a ball cost $1.10; the bat is $1.00 more than the ball. How much is the ball?", "0.05", "sum-diff"),
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the bat?", "1.05", "sum-diff"),
    ("A hammer and a nail cost $21 together. The hammer costs $20 more than the nail. How much is the nail?", "0.5", "sum-diff"),
    ("A phone and a case cost $110 in total. The phone costs $100 more than the case. How much is the case?", "5", "sum-diff"),
    ("A widget and a gadget cost $2.50 together. The widget costs $2.00 more than the gadget. How much is the gadget?", "0.25", "sum-diff"),
    ("The sum of two numbers is 30 and their difference is 4. What is the larger number?", "17", "sum-diff"),
    ("The sum of two numbers is 30 and their difference is 4. What is the smaller number?", "13", "sum-diff"),
    # reverse-pct — the "$48" class, solved exactly
    ("A shirt costs $40 after a 20% discount. What was the original price?", "50", "reverse-pct"),
    ("After a 15% discount, a lamp costs $68. What was the original price?", "80", "reverse-pct"),
    ("A jacket was discounted 30% and now costs $84. What was the original price?", "120", "reverse-pct"),
    ("After a 25% raise, my salary is $75,000. What was my salary before the raise?", "60000", "reverse-pct"),
    ("A number increased by 30% is 78. What is the number?", "60", "reverse-pct"),
    # shift
    ("A number decreased by 12 is 39. What is the number?", "51", "shift"),
    ("A number increased by 17 is 60. What is the number?", "43", "shift"),
    # combo
    ("Movie tickets cost $9 for kids and $14 for adults. What do 3 kids and 2 adults pay in total?", "55", "combo"),
    ("Adult tickets cost $12 and child tickets cost $7. What do 2 adult and 3 child tickets cost in total?", "45", "combo"),
]

# Shapes we must NOT touch: set-logic, work-rate, exponential, multi-step, partial
# parses, unit mismatches, number-words, negatives, and plain non-quantitative text.
DECLINE = [
    "Emma has 5 brothers. Each brother has 3 sisters. How many sisters does Emma have?",
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
    "Tom has 4 sisters. Each of his sisters has 2 brothers. How many brothers does Tom have?",
    "If 7 workers dig 7 holes in 7 hours, how long do 14 workers need to dig 14 holes?",
    "It takes 5 machines 5 minutes to make 5 widgets. How long would it take 100 machines to make 100 widgets?",
    "A bacteria colony doubles every hour and fills the dish at hour 24. At what hour was the dish half full?",
    "A patch of lily pads doubles in size every day. It takes 48 days to cover the whole lake. How many days to cover half the lake?",
    "A snail climbs 3 meters up a wall each day and slips back 2 meters each night. The wall is 10 meters tall. How many days does it take to reach the top?",
    "If a chicken and a half lays an egg and a half in a day and a half, how many eggs does one chicken lay in one day?",
    "A farmer has chickens and cows, 20 heads and 56 legs in total. How many cows are there?",
    "A rope is cut into two pieces. One piece is 3 times as long as the other. The rope was 24 m. How long is the shorter piece?",
    "Three consecutive integers add up to 51. What is the smallest one?",
    "A brick weighs 1 kg plus half a brick. How much does the brick weigh?",
    "A pen costs $1.35. A notebook costs twice as much as the pen. How much do two notebooks and one pen cost together?",
    # partial parses / extra quantities — the v1.1.1 lesson, template edition
    "A crate holds 1,728 units. 56 crates arrive and 3 are damaged. How many undamaged units are left?",
    "Each box weighs 23.7 kg. What is the total weight of 41 boxes plus a 12 kg pallet?",
    "Each box holds 40 cans. How many cans in half of 10 boxes?",
    # unit mismatch across declaration and question
    "A server processes 3,408 requests per minute. How many requests in 2 hours?",
    # noun mismatch
    "Each pallet holds 3,672 cans. How many cans are on 38 trucks?",
    # money mismatch inside the pair
    "A bat and a ball cost $1.10 total. The bat costs 1.00 more than the ball. How much is the ball?",
    # gap >= total is a trick, not a shape
    "A bat and a ball cost $1.10 total. The bat costs $1.20 more than the ball. How much is the ball?",
    # two-step percent, forward percent, change-making, negatives
    "A price rises 10% and then falls 10%. What percent of the original price is it now?",
    "What is 20% off 50?",
    "I paid with a $20 bill for a $13.45 purchase. How much change should I get?",
    "The temperature was -8 degrees Celsius and rose by 15 degrees. What is the temperature now?",
    # not quantitative at all / wrong number count
    "What is the capital of Australia?",
    "How many moons does Mars have?",
    "What is 47 times 19?",
    "Convert 5 feet 4 inches to centimeters.",
    "",
]


def main():
    ok = 0
    for q, want, want_name in PASS:
        got = solve(q)
        good = got is not None and got[0] == want and got[1] == want_name
        ok += good
        print(f"{'ok ' if good else 'FAIL'} {want_name:<12} {q[:58]:<58} -> {got}  (want {want!r})")
    for q in DECLINE:
        got = solve(q)
        good = got is None
        ok += good
        print(f"{'ok ' if good else 'FAIL'} decline      {q[:58]:<58} -> {got}")
    total = len(PASS) + len(DECLINE)
    print("-" * 100)
    print(f"PASS  {ok}/{total}")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
