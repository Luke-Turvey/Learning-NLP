import numpy as np

def min_edit_distance(string1,string2,del_cost=1,ins_cost=1,sub_cost=1):
    n = len(string1)
    m = len(string2)

    D = np.zeros((n+1,m+1),dtype=int)
    for i in range(0,n+1):
        D[i,0] = i
    for j in range(0,m+1):
        D[0,j] = j

    for row in range(1,n+1):
        for column in range(1,m+1):
            delete_value = D[row-1,column] + del_cost
            insertion_value = D[row,column-1] + ins_cost
            if string1[row-1] == string2[column-1]:
                substitution_value = D[row-1,column-1]
            else:
                substitution_value = D[row-1,column-1]+sub_cost

            D[row,column] = min(delete_value,insertion_value,substitution_value)

    return D[n,m]

print(min_edit_distance("processing","insisting"))
