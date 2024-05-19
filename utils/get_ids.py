def rest_ids(done_dir, n_rna, maintain_solved=False, init_id_list=None):
    if init_id_list is None:
        totel_id_list = list(range(1, n_rna+1))
    else:
        totel_id_list = init_id_list
    done_id_list = []

    f = open(done_dir)
    iter_f = iter(f)
    for line in iter_f:
        done_id = line.split(' ')[1]
        done_id = done_id.split('_')[1]
        done_id = int(done_id)
        done_id_list.append(done_id)

    done_id_list = list(set(done_id_list))

    if not maintain_solved:
        rest_id_list = [id for id in totel_id_list if id not in done_id_list]
        print("Maintain sloved puzzles.")
    else:
        rest_id_list = totel_id_list
        print("Load all puzzles.")

    # print("Load {} puzzles".format(len(rest_id_list)))

    return rest_id_list, done_id_list


def split_ids(rest_id_list, batch_size):
    id_batch_list = BatchSampler(SubsetRandomSampler(rest_id_list), batch_size, False)
    id_batch_list = list(id_batch_list)
    return id_batch_list



if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from torch.utils.data import BatchSampler, SubsetRandomSampler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--done_dir", type=Path, help="Path to done ids."
    )
    parser.add_argument(
        "--n_rna", type=int, help="The totel Number of the RNAs"
    )
    parser.add_argument(
        "--batch_size", type=int, help="The number of structure trained in one epoch."
    )
    parser.add_argument(
        "--save_dir", type=Path, help="Path to id batchs."
    )

    parser.add_argument(
        "--forbid_id", type=int, help="Path to id batchs.", default=-1
    )

    parser.add_argument(
        "--maintain_solved", action='store_true', help="Path to id batchs.", 
    )

    parser.add_argument(
        "--set_init_ids", action='store_true', help="Path to id batchs.", 
    )





    args = parser.parse_args()

    forbid_id = args.forbid_id

    init_ids = None
    if args.set_init_ids:
        print("Set initial ids.")
        init_ids = [33, 35, 52,59,62,65,66,67,71,72,77,79,83]

    rest_id_list, done_id_list = rest_ids(args.done_dir, args.n_rna, args.maintain_solved, init_ids)
    # print(rest_id_list, done_id_list)
    id_batch_list = split_ids(rest_id_list, args.batch_size)
    f = open(args.save_dir, "+w")
    for id_batch in id_batch_list:
        id_str = ''
        for id in id_batch:
            if id == forbid_id:
                pass
            else:
                id_str += str(id)
                id_str += ','
        f.write(id_str)
        f.write(' ')
        f.flush()
    f.close()
    # print(id_batch_list)

    

    

