tom1 = open("wiedźmin/1Sapkowski_Andrzej_-_Wied_378_min_-_Sezon_burz.txt",'r',encoding='latin-1')
tom1read = tom1.read()
tom1.close()

tom2 = open("wiedźmin/2Ostatnie__380_yczenie.txt",'r',encoding='latin-1')
tom2read = tom2.read()
tom2.close()

tom3 = open('wiedźmin/3Miecz_przeznaczenia.txt','r',encoding='latin-1')
tom3read = tom3.read()
tom3.close()

tom4 = open('wiedźmin/4Krew_elf_243_w.txt','r',encoding='latin-1')
tom4read = tom4.read()
tom4.close()

tom5 = open('wiedźmin/5Czas_pogardy.txt','r',encoding='latin-1')
tom5read = tom5.read()
tom5.close()

tom6 = open("wiedźmin/6Andrzej_Sapkowski_-_Chrzest_Ognia.txt",'r',encoding='latin-1')
tom6read = tom6.read()
tom6.close()

tom7 = open("wiedźmin/7Wie_380_a_jask_243__322_ki.txt",'r',encoding='latin-1')
tom7read = tom7.read()
tom7.close()

tom8 = open("wiedźmin/8Pani_jeziora.txt",'r',encoding='latin-1')
tom8read = tom8.read()
tom8.close()

wszystkieTomy = tom1read + tom2read + tom3read + tom4read + tom5read + tom6read + tom7read + tom8read

writeAll = open("Wiedzmin.txt","w",encoding='latin-1')
writeAll.write(wszystkieTomy)
writeAll.close()
