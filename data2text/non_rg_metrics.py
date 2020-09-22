from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
from collections import namedtuple
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from text2num import text2num, NumberException
from utils import divide_or_const

try:
    range = xrange
except:
    pass

Item = namedtuple("Item", ["number", "label", "entry"])

full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
 'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
 'Miami Heat', 'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
 'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards', 'Dallas Mavericks',
 'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets', 'Los Angeles Clippers',
 'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves', 'New Orleans Pelicans',
 'Oklahoma City Thunder', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings',
 'San Antonio Spurs', 'Utah Jazz']

cities, teams = set(), set()
ec = {} # equivalence classes
for team in full_names:
    pieces = team.split()
    if len(pieces) == 2:
        ec[team] = [pieces[0], pieces[1]]
        cities.add(pieces[0])
        teams.add(pieces[1])
    elif pieces[0] == "Portland": # only 2-word team
        ec[team] = [pieces[0], " ".join(pieces[1:])]
        cities.add(pieces[0])
        teams.add(" ".join(pieces[1:]))
    else: # must be a 2-word City
        ec[team] = [" ".join(pieces[:2]), pieces[2]]
        cities.add(" ".join(pieces[:2]))
        teams.add(pieces[2])


def entry_eq(e1, e2):
    if e1 in cities or e1 in teams or e2 in cities or e2 in teams:
        return e1 == e2 or any(e1 in fullname and e2 in fullname for fullname in full_names)
    else:
        return e1 in e2 or e2 in e1

def item_eq(t1, t2):
    return t1.number == t2.number and t1.label == t2.label and entry_eq(t1.entry, t2.entry)

def dedup_items(item_list):
    """
    this will be inefficient but who cares
    """
    ret = []
    for i, t_i in enumerate(item_list):
        for j in range(i):
            t_j = item_list[j]
            if item_eq(t_i, t_j):
                break
        else:
            ret.append(t_i)
    return ret

def get_items(fi):
    def process_item(item):
        try:
            item[0] = int(item[0])
        except ValueError:
            try:
                item[0] = text2num(item[0])
            except NumberException:
                pass
        return Item(*item)

    with open(fi) as f:
        return list(map(
            lambda line: dedup_items(list(filter(
                lambda item: isinstance(item.number, int),
                map(lambda s: process_item(s.split('|')),
                    line.strip().split())))),
            f))

def calc_precrec(gold_items, pred_items, itemwise_outfile=None):
    total_tp, total_pred, total_gold = 0, 0, 0
    for gold_item_list, pred_item_list in zip(gold_items, pred_items):
        tp = sum(1 for pred_item in pred_item_list if any(
                 item_eq(pred_item, gold_item) for gold_item in gold_item_list))
        total_tp += tp
        total_pred += len(pred_item_list)
        total_gold += len(gold_item_list)
        if itemwise_outfile is not None:
            print('{tp}\t{prec:.6f}\t{rec:.6f}'.format(
                      tp=tp,
                      prec=divide_or_const(tp, len(pred_item_list)),
                      rec=divide_or_const(tp, len(gold_item_list))),
                  file=itemwise_outfile)

    avg_prec = divide_or_const(total_tp, total_pred)
    avg_rec = divide_or_const(total_tp, total_gold)
    print("total_tp: {} total_pred: {} total_gold: {}".format(
        total_tp, total_pred, total_gold))
    print("prec: {} rec: {}".format(avg_prec, avg_rec))
    return avg_prec, avg_rec

def norm_dld(l1, l2):
    ascii_start = 0
    # make a string for l1
    # all items are unique...
    s1 = ''.join((chr(ascii_start+i) for i in range(len(l1))))
    s2 = ''
    next_char = ascii_start + len(s1)
    for j in range(len(l2)):
        found = None
        #next_char = chr(ascii_start+len(s1)+j)
        for k in range(len(l1)):
            if item_eq(l2[j], l1[k]):
                found = s1[k]
                #next_char = s1[k]
                break
        if found is None:
            s2 += chr(next_char)
            next_char += 1
            assert next_char <= 128
        else:
            s2 += found
    # return 1- , since this thing gives 0 to perfect matches etc
    return 1.0-normalized_damerau_levenshtein_distance(s1, s2)

def calc_dld(gold_items, pred_items):
    total_score = 0
    for gold_item_list, pred_item_list in zip(gold_items, pred_items):
        total_score += norm_dld(pred_item_list, gold_item_list)
    avg_score = total_score / len(pred_items)
    print("avg score:", avg_score)
    return avg_score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('gold_file')
    argparser.add_argument('pred_file')
    args = argparser.parse_args()
    gold_items, pred_items = map(
        get_items, (args.gold_file, args.pred_file))
    assert len(gold_items) == len(pred_items), \
        "len(gold) = {}, len(pred) = {}".format(
            len(gold_items), len(pred_items))
    calc_precrec(gold_items, pred_items)
    calc_dld(gold_items, pred_items)
