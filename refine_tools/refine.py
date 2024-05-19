from copy import deepcopy

import RNA
import numpy as np
import random
import os
import pickle
from .refine_moves import *


def seq_to_struct(pred_solution):
    '''Returns predicted structure and energy given a sequence.'''
    # p = Popen(['RNAfold'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    # output = p.communicate(input=pred_solution)[0]
    output = RNA.fold(pred_solution)[0]
    # pred_struct = re.split('\s+| \(?\s?', output)[1]
    pred_struct = re.split('\s+| \(?\s?', output)[0]
    return pred_struct


def refine_check_answer(dot_bracket, pred_solution):
  pred_struct = seq_to_struct(pred_solution)
  correct = 0
  for i in range(len(pred_struct)):
    if pred_struct[i] == dot_bracket[i]:
      correct += 1
  # Compute long-range erroneous pairs
  pred_pairs = bracket_to_bonds(pred_struct)
  true_pairs = bracket_to_bonds(dot_bracket)
  bad_pairs = []
  for i in pred_pairs:
    if i not in true_pairs:
      bad_pairs.append(i)
  missing_pairs = []
  for i in true_pairs:
    if i not in pred_pairs:
      missing_pairs.append(i)
  return correct / float(len(dot_bracket)), true_pairs, bad_pairs, missing_pairs

def refine_check_answer_dist(dot_bracket, pred_solution):
  pred_struct = seq_to_struct(pred_solution)
  correct = 0
  for i in range(len(pred_struct)):
    if pred_struct[i] == dot_bracket[i]:
      correct += 1
  # Compute long-range erroneous pairs
  pred_pairs = bracket_to_bonds(pred_struct)
  true_pairs = bracket_to_bonds(dot_bracket)
  bad_pairs = []
  for i in pred_pairs:
    if i not in true_pairs:
      bad_pairs.append(i)
  missing_pairs = []
  for i in true_pairs:
    if i not in pred_pairs:
      missing_pairs.append(i)
  return correct / float(len(dot_bracket)), true_pairs, bad_pairs, missing_pairs, len(dot_bracket) - correct


def bracket_to_bonds(structure):
    bonds = [None]*len(structure)
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            bonds[i] = j
            bonds[j] = i
    reshaped_bonds = []
    for i in range(len(bonds)):
        if bonds[i] != None:
            pair = [i, bonds[i]]
            if pair not in reshaped_bonds and [bonds[i], i] not in reshaped_bonds:
                reshaped_bonds.append(pair)
    return reshaped_bonds


def refine(dataset, n_trajs, n_steps, move_set):
    refined_data = []
    for puzzle in dataset:
        # if puzzle_name != '-1' and puzzle[0] != puzzle_name:
        #     continue
        if puzzle[-1] < 1:
            # print('Trying to refine %s'%(puzzle[0]))
            dot_bracket = puzzle[1]
            input_solution = puzzle[2]
            accuracy, _, _, _ = refine_check_answer(dot_bracket, input_solution)
            for traj in range(n_trajs):
                solution = deepcopy(input_solution)
                # print('Trajectory %d'%(traj))
                if accuracy == 1:
                    # print('Found a valid solution')
                    break
                # solution = deepcopy(input_solution)
                move_traj = np.random.choice(move_set, n_steps, replace=True)
                for move in move_traj:
                    output, solution = move.apply(dot_bracket, solution)
                    try:
                        output, solution = move.apply(dot_bracket, solution)
                    except:
                        break
                    new_accuracy, _, _ = output
                    if new_accuracy > accuracy:
                        # print('Found better accuracy threshold of %f'%(new_accuracy))
                        # print(solution)
                        accuracy = new_accuracy
                        input_solution = solution
                        break
            puzzle_refine = [puzzle[0], dot_bracket, solution, accuracy]
            refined_data.append(puzzle_refine)
        else:
            refined_data.append(puzzle)

    return refined_data


def sent_refine_single(dot_bracket, input_solution, n_trajs, n_steps, move_set):
    init_real_dotB = RNA.fold(input_solution)[0]
    init_dist = RNA.hamming(init_real_dotB, dot_bracket)
    accuracy, _, _, _ = refine_check_answer(dot_bracket, input_solution)
    is_refined = False
    for traj in range(n_trajs):
        # solution = deepcopy(input_solution)
        # print('Trajectory %d'%(traj))
        if accuracy == 1:
            is_refined = True
            # print('Found a valid solution')
            break
        solution = deepcopy(input_solution)
        move_traj = np.random.choice(move_set, n_steps, replace=True)
        for move in move_traj:
            output, solution = move.apply(dot_bracket, solution)
            try:
                output, solution = move.apply(dot_bracket, solution)
            except:
                break
            new_accuracy, _, _ = output
            if new_accuracy > accuracy:
                is_refined = True
                # print('Found better accuracy threshold of %f'%(new_accuracy))
                # print(solution)
                accuracy = new_accuracy
                input_solution = solution
                break

    real_dotB = RNA.fold(input_solution)[0]
    real_dist = RNA.hamming(real_dotB, dot_bracket)
    if real_dist > init_dist:
        print('error 1')
    return input_solution, int((1 - accuracy) * len(dot_bracket)), is_refined


def refine_from_dataset(dataset, n_trajs, n_steps):

    move_set = [GoodPairsMove(), BadPairsMove(), MissingPairsMove(), BoostMove()]

    refine_data = refine(dataset, n_trajs, n_steps, move_set)

    return refine_data


if __name__ == "__main__":
    dataset = [
        # [56, '((.(..(.(....).(....).)..).(....).))', 'UAUCAGGCAGAGCUUCGAAAGUCUCGACAGAGGGUA', 0]
    ]

    puzzle_name = []

    data_dir = './best_dict.pkl'

    f = open(data_dir, 'rb')
    best_dict = pickle.load(f)

    for key in best_dict.keys():
        name = int(key)
        data = best_dict[key]
        data = [name, str(data[2]).split('\n')[0], str(data[1]).split('\n')[0], 0]
        puzzle_name.append(name)
        dataset.append(data)

    move_set = [GoodPairsMove(),BadPairsMove(),MissingPairsMove(),BoostMove()]

    n_trajs = 300
    n_steps = 30

    # puzzle_name = 56

    out_path = 'test.pkl'

    dotB_2004 = RNA.fold(dataset[0][2])[0]

    RNA_param = '../RNA_param/rna_turner1999.par'

    RNA.read_parameter_file(RNA_param)

    # dotB_1999 = RNA.fold(dataset[0][2])[0]
    #
    # dist = RNA.hamming(dotB_1999, dotB_2004)
    #
    # refine_data = refine(dataset, n_trajs, n_steps, move_set)

    dotB = '((.(..(.(....).(....).)..).(....).))'
    seq_base = 'UAUCAGGCAGAGCUUCGAAAGUCUCGACAGAGGGUA'
    result = sent_refine_single(dotB, seq_base, n_trajs, n_steps, move_set)

    print(1)