


with open(r"K:\programme\python\Projekte\woerter\data\word_list_english_uppercase_spell_checked.txt","r") as dat:
    
    maxdict=dict()
    words=[]
    for i,line in enumerate(dat):
        line=line.strip()
        words=words[:len(line)]
        current=(i,line)
        for j in range(len(words)):
            if line[j]!=words[j][1][j]:
                words[j]=current
            else:
                length=(i-words[j][0])
                #print(length)
                if maxdict.get(j,[-1])[0]<length:
                    maxdict[j]=length,(words[j][1],line)

        words+=[current]*(len(line)-len(words))
    print(maxdict)
    for i,(length,(s,e)) in maxdict.items():
        print(s[:i+1],length,s,e)