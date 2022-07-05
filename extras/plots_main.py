sample=50
fig, axes = plt.subplots()
fig.set_size_inches(6,6)

axes.set_xlim(0,ngen)
axes.set_ylim(0,1)
axes.set_yticks(np.arange(0,1.1,0.1))
axes.set_xlabel("Generation",size="16")
axes.set_ylabel("Mean average score",size="16")

food_list = [10,20,30]

cumF = [[0]*ngen for _ in range(3)]

for food_ind in range(len(food_list)):
    print(food_list[food_ind])
    for x in range(sample):
        env = Environment()
        env.gen_pop(npop)
        for gen in range(ngen):
            
            kwarg = {'show_steps':True if gen == 0 else False}
            if not env.evolution(axes,show_steps=False):
                print("E' sopravvissuto un solo individuo: la popolazione è estinta.")
                break
            else:
                cumF[food_ind][gen] += mean([ind.score() for ind in env.pop])
                env.distribute(nfood=food_list[food_ind])
                env.gen_pop(nind = npop, mut_prob = mut_p)

axes.set_title("Pop size: " +str(npop)+ ", Sample size: " +str(sample),size="18")
axes.axhline(0.5,linestyle="--",label="Random population",c="magenta")
axes.scatter(range(ngen), [x/sample for x in cumF[0]], marker='.', c='blue', clip_on=False,label=str(food_list[0])+" food")
axes.scatter(range(ngen), [x/sample for x in cumF[1]], marker='.', c='red', clip_on=False,label=str(food_list[1])+" food")
axes.scatter(range(ngen), [x/sample for x in cumF[2]], marker='.', c='green', clip_on=False,label=str(food_list[2])+" food")
axes.legend(loc='lower left', fontsize="13")
plt.show()