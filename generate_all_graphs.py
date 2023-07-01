#! /usr/bin/python -u
import glob, os
import datetime, time
import argparse as ap
from tools import *
from graph_tools import *


''' Load and return messages in fileName '''
def comments_from_file(fileName):
    content = zload(fileName)
    comments = [c for c in content]
    return comments

''' Load all comments from part p '''
def load_part_comments(p):
    X, Y = [], []
    non_abuse_file = glob.glob("10-Split/*0.*.pkl.gz-%s*" % p)[0]
    non_abuse = comments_from_file(non_abuse_file)
    abuse_file = glob.glob("10-Split/*1.*.pkl.gz-%s*" % p)[0]
    abuse = comments_from_file(abuse_file)
    Y.extend([0 for v in range(len(non_abuse))])
    Y.extend([1 for v in range(len(abuse))])
    X.extend(non_abuse)
    X.extend(abuse)
    return X, Y

''' Split raw message into timestamp, user id, message type, text '''
def split_raw_message(raw_message):
    text = raw_message[-1]
    mtype = raw_message[-2]
    if raw_message[2] == b'message':
        if raw_message[0] > 1400000000:
            return raw_message[0], raw_message[1], mtype, text
        return raw_message[1], raw_message[0], mtype, text
    elif raw_message[2] == b'chatMessage':
        return raw_message[0], raw_message[1], mtype, text

''' Get the data corresponding to part number p and indice i. '''
def get_all_data(p, i, part_context, part_comments):
    
    # Load raw message
    comments, labels = part_comments
    label = labels[i]
    context = part_context[i]
    raw_message = comments[i]

    mtimestamp, user_id, mtype, msg = split_raw_message(raw_message)
    raw_message = (mtimestamp, user_id, mtype, msg)
    return raw_message, label, context

def normalize_context(context):
    ''' (context) -> context

    Normalize context depending on the message type
    The goal is to return context in a consistent format.
    No matter the message type output of this function is
    pre, target_message, post
    Messages in pre and post are time, uid, msg with time format f.
    target_message is also in that format so we can concatenate all elements
    of the return tuple to get the history.

    '''
    f = '%Y-%m-%d/%H:%M:%S'
    pre, target, post, source = context
    if target[2] == b'message':
        t_time, t_uid, t_message = target[0], target[1], target[3]
        if t_uid > 1400000000: 
            t_time, t_uid = t_uid, t_time
        target_out = (time.strftime(f, time.localtime(t_time)), t_uid, t_message)
        pre_out = []
        for row in pre:
            t, u, m = row
            t = time.strftime(f, time.localtime(t))
            pre_out.append((t, u, m))
        post_out = []
        for row in post:
            t, u, m = row
            t = time.strftime(f, time.localtime(t))
            post_out.append((t, u, m))
    else:
        t_uid, t_time, t_message = target[1], target[0], target[3]
        target_out = (time.strftime(f, time.localtime(t_time)), t_uid, t_message)
        pre_out = []
        for row in pre:
            u, t, m = row[1], row[0], row[3]
            t = t.strftime(f)
            pre_out.append((t, u, m))
        post_out = []
        for row in post:
            u, t, m = row[1], row[0], row[3]
            t = t.strftime(f)
            post_out.append((t, u, m))
    return (pre_out, target_out, post_out, source)

def memdrop_graph(name, p, i, rows, label, raw_message, source, target_row, window_size=10, distrib='spread', directed=False, vs=None):

    if name.startswith('full'):
        g, g_classic, g_pos, g_neg = build_graphs(rows, window_size=window_size, distrib=distrib, directed=directed)
        g['label'] = label
        g_classic['label'] = label
        g_pos['label'] = label
        g_neg['label'] = label
        g['p'] = p
        g_classic['p'] = p
        g_pos['p'] = p
        g_neg['p'] = p
        g['i'] = i
        g_classic['i'] = i
        g_pos['i'] = i
        g_neg['i'] = i
        g['g_name'] = name
        g_classic['g_name'] = name
        g_pos['g_name'] = name
        g_neg['g_name'] = name
        g['target_uid'] = rows[target_row][1]
        g_classic['target_uid'] = rows[target_row][1]
        g_pos['target_uid'] = rows[target_row][1]
        g_neg['target_uid'] = rows[target_row][1]
        fname = "GRAPHES/%s/%s/%s.graphml" % (p, i, name)
        return fname, g, g_classic, g_pos, g_neg
    else:
        g, g_classic, g_pos, g_neg = build_graphs(rows, window_size=window_size, distrib=distrib, directed=directed)
        g['g_name'] = name
        g_classic['g_name'] = name
        g_pos['g_name'] = name
        g_neg['g_name'] = name
        fname = "GRAPHES/%s/%s/%s.graphml" % (p, i, name)
        return fname, g, g_classic, g_pos, g_neg

def drop_dir(p, i):
    if not os.path.isdir("GRAPHES/%s/%s" % (p, i)):
        os.makedirs("GRAPHES/%s/%s" % (p, i))

# chcp 65001 
if __name__ == '__main__':
    
    p = ap.ArgumentParser()
    p.add_argument("-c", "--context-size", type = int, default = 800)
    p.add_argument("-w", "--window-size", type = int, default = 10)
    p.add_argument("-d", "--directed", action="store_true")
    p.add_argument("-s", "--distribution_strategy", type = str, default = 'spread')
    args = p.parse_args()

    cpt = 0

    for part in split_gen():
        part_context = zload("context/extra-%s.pkl.gz" % part)
        part_comments = load_part_comments(part)

        for i in range(len(part_context)):
            if cpt <= 1386:
                cpt += 1
            else:
                raw_message, label, context = get_all_data(p, i, part_context, part_comments)
                context = normalize_context(context)

                pre, target, post, source = context
                half_context = args.context_size // 2
                pre = pre[-half_context:]
                post = post[:half_context]

                rows = list(pre) + [list(target)] + list(post)
                target_row = len(pre)

                suffix = "_cs%s_ws%s_d%s_ds%s" % (args.context_size, args.window_size, args.directed, args.distribution_strategy)

                fname, g, g_classic, g_pos, g_neg = memdrop_graph("full" + suffix, p, i, rows, label, raw_message, source, target_row, directed=args.directed, window_size = args.window_size, distrib=args.distribution_strategy)
                #pre_rows = list(pre) + [list(target)]
                #fnameb, gb, gb_classic, gb_pos, gb_neg = memdrop_graph("before" + suffix, p, i, pre_rows, label, raw_message, source, target_row, directed=args.directed, window_size = args.window_size, distrib=args.distribution_strategy, vs=g.vs)

                #post_rows = [list(target)] + list(post)
                #fnamea, ga, ga_classic, ga_pos, ga_neg = memdrop_graph("after" + suffix, p, i, post_rows, label, raw_message, source, target_row, directed=args.directed, window_size = args.window_size, distrib=args.distribution_strategy, vs=g.vs)

                #Write to disk
                g.save("GRAPHES_context800/signe/full_%s.graphml" % cpt)
                g_classic.save("GRAPHES_context800/non_signe/full_%s.graphml" % cpt)
                g_pos.save("GRAPHES_context800/positive/full_%s.graphml" % cpt)
                g_neg.save("GRAPHES_context800/negative/full_%s.graphml" % cpt)

                #Write to disk
                '''g.save("GRAPHES/signe/full_%s.graphml" % cpt)
                gb['label'] = g['label']
                gb.save("GRAPHES/signe/before_%s.graphml" % cpt)
                ga['label'] = g['label']
                ga.save("GRAPHES/signe/after_%s.graphml" % cpt)

                g_classic.save("GRAPHES/non_signe/full_%s.graphml" % cpt)
                gb_classic['label'] = g['label']
                gb_classic.save("GRAPHES/non_signe/before_%s.graphml" % cpt)
                ga_classic['label'] = g['label']
                ga_classic.save("GRAPHES/non_signe/after_%s.graphml" % cpt)

                g_pos.save("GRAPHES/positive/full_%s.graphml" % cpt)
                gb_pos['label'] = g['label']
                gb_pos.save("GRAPHES/positive/before_%s.graphml" % cpt)
                ga_pos['label'] = g['label']
                ga_pos.save("GRAPHES/positive/after_%s.graphml" % cpt)

                g_neg.save("GRAPHES/negative/full_%s.graphml" % cpt)
                gb_neg['label'] = g['label']
                gb_neg.save("GRAPHES/negative/before_%s.graphml" % cpt)
                ga_neg['label'] = g['label']
                ga_neg.save("GRAPHES/negative/after_%s.graphml" % cpt)'''
                cpt += 1
                print ("%s/%s" % (cpt, len(part_context)))
