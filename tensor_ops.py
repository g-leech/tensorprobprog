import random
import torch as t
import tpp_trace as tra
import utils as u
import math


def combine_tensors(tensors_dict, hasPlates=False) :    
    """
    Our index-aware tensor product. Takes indexed log_prob tensors as input,
    sums out all indices, and returns dict with one scalar, the loss.
    
    arg:
        - **tensors_dict (dict):** dict of log_prob tensors to combine 
    """
    # Take all indices I in the tensors, make a unified set 
    all_names = get_all_names(tensors_dict)
    has_no_pos(all_names)
    k_dims = get_k_names(all_names)
    
    # For each k dim, get tensors that depend on k and reduce
    while k_dims :
        random_index = random.randrange(len(k_dims))
        dim = k_dims.pop(random_index)
        tensors_dict = reduce_by_dim(tensors_dict, dim, hasPlates)
    
    return tensors_dict


# Check positional dimensions are already gone
def has_no_pos(names) :
    assert(all([tra.pos_name("") not in el \
                for el in names]))

    
def get_all_names(d) :
    """
    Use Torch NamedTensor to enumerate dimensions in tensor dict 
       
    arg:
        - **d (dict):** dict of NamedTensors
    """
    dims = list(d.keys()) 
    all_names = [d[dim].names for dim in dims]
    flattened = list(set([item for t in all_names \
                     for item in t]))
    return [n for n in flattened if n]


def get_k_names(names, plates=False) :
    """
    Include only index dimensions, i.e. the "k_"
    
    arg:
        - **names (list):** list of dimension names 
        - **plates (bool):** whether to traverse plate dimensions
    """
    if not plates :
        names = [name for name in names \
                  if tra.k_dim_name("") in name]
        
    return names


def get_plate_names(names) :
    plate_str = tra.plate_dim_names("", 1)[0]
    
    return [name for name in names if plate_str in name]


def clear_user_dims(d, names) :
    """
        sum out non-index dimensions
    """
    user_dims = [ name for name in names \
                    if tra.pos_name("") in name ]
    
    for user_dim in user_dims :
        for k, T in d.items() :
            d[k] = T.sum(user_dim)
    
    return d


def reduce_by_dim(d, dim, plates=False):
    """
    The central tensor product over one index dimension.
    Assume log space, so the tensor product is simply a sum of log_probs, 
    a logsumexp to remove the index dim, and normalising by the index size.
    
    arg:
        - **d (dict):** dict of log_prob tensors to combine 
        - **dim (str):** (name of the) dimension to reduce over and sum out 
    """
    if not plates:
        assert( tra.k_dim_name("") in dim )
    
    other_tensors, i_tensors = get_dependent_factors(d, dim)
    
    all_names = get_all_names(i_tensors)
    plate_dims = get_plate_names(all_names)
    k_dims = get_k_names(all_names)
    
    # use torch names to get all dims in same order
    for k, tensor in i_tensors.items() :
        i_tensors[k] = tensor.align_to(*k_dims, *plate_dims)
    
    # TODO: Can just pop the first i_tensor?
    T = 0

    # multiply (as sum of logs)
    for k, tensor in i_tensors.items() :
        T = T + tensor
    
    nk = T.size(dim)
    # 6. sum out dim
    T = t.logsumexp(T, dim) - math.log(nk)
    # 7. put it back
    other_tensors[dim] = T
    
    return other_tensors


def get_dependent_factors(d, i):
    """
    For a given index dimension i, finds the factors T∣K_i that depend on K_i.
    arg:
        - **d (dict):** dict of log_prob tensors to combine 
        - **i (str):** (name of the) index dimension to find tensors dependent on
    """
    # TODO: consider lists instead
    dependents = { k: tensor for k, tensor in d.items() \
                    if i in tensor.names }
    nondependents = { k: tensor for k, tensor in d.items() \
                        if i not in tensor.names }
    
    return nondependents, dependents


# Using implementation detail of dict in 3.6+
# TODO: think about edge cases
def get_ordered_tensors(d) :
    return d

def get_ordered_plate_names(d) :
    all_names = get_all_names(d)
    
    return [n for n in all_names if "plate_" in n][::-1]

def has_plate(tensor, plate_name) :
    return plate_name in tensor.names


def combine_over_plates(lps) :
    tensor_dict = get_ordered_tensors(lps) #tensors ordered by generation (P)
    plate_names = get_ordered_plate_names(lps) #plates ordered by appearance (P)

    #go backwards through plates
    for plate in plate_names[::-1]: 
        print(plate)
        # gather all tensors with that plate
        plate_tensors = {k:tensor for k, tensor in tensor_dict.items() \
                           if has_plate(tensor, plate)}
        #print(plate_tensors)
        #remove tensors with that plate from the global list,
        other_tensors = [ tensor for tensor in tensor_dict.values() \
                           if not has_plate(tensor, plate) ]
        #print(other_tensors)
        # combine those tensors using code we already have, 
        # which shouldn't need to know about plates
        d = combine_tensors(plate_tensors, hasPlates=True) 
        print(d)
        #sum out the current plate
        tensor = undict(d) \
                    .sum(plate)
        print(tensor)
        # put the tensor back in the global list for further reduction
        other_tensors.append(tensor)
    
    return other_tensors[0].sum()


def undict(d) :
    assert(len(d.keys()) == 1)
    key = next(iter(d))

    return d[key]


if __name__ == "__main__" :
    kappa = 2
    n = 2
    data = {} # {"a": 4}
    tr = tra.sample_and_eval(chain_dist, draws=kappa, nProtected=n, data=data)
    tensors = tr.trace.out_dicts['log_prob']

    print(combine_tensors(tensors))