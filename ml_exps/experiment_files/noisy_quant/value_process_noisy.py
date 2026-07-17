import pickle

model = 'resnet32'
mega = 1024 * 1024
for bit in range(3, 6):
    acc = 0
    time = 0
    mem = 0
    act = 0
    overhead = 0
    overhead_all = 0
    for seed in range(5):
        with open('experiment_res/noisy_res_' + str(model) + '_seed_' + str(seed) + '_bit_' + str(bit) + '.pkl', 'rb') as file:
            data = pickle.load(file)
            print(data)
            acc = acc + data['acc']/5
            time = time + data['time']/5
            mem = mem + data['mem']/5
        #break
            #act = act + data['act']/10
            #overhead = overhead + data['overhead']/10
            #overhead_all = overhead_all + data['overhead']
    
    print('Bit', bit)
    print('acc', round(acc,2))
    print('time', round(time,2))
    print('mem', round(mem/mega,2)) 
    #print('act', round(act/mega,2))
    #print('overhead', round(overhead/mega,2))
    #print('overhead_per_image', round((overhead_all/mega)/10000,5))
