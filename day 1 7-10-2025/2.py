def calc_f1_score( tp,fp,fn):
    if( type(tp)!=int or type(fp)!=int or type(fn)!=int):
        print("Input must be integer")
        return
    elif tp<=0 or fp<=0 or fn<=0:
        print("Input must be positive integer")
        return

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1_score = 2*precision*recall/(precision+recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {F1_score}")

print(20*"=")
calc_f1_score(70,10,20)
print(20*"=")
calc_f1_score(70,-10,20)
print(20*"=")
calc_f1_score(70,10.5,20)
print(20*"=")

