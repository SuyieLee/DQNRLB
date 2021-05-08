import matplotlib.pyplot as plt


# num = [24583.0, 70852.0, 112824.0, 143348.0, 165866.0, 187869.0, 221920.0, 243795.0, 262907.0, 273652.0, 286010.0, 296202.0, 315148.0, 324220.0, 338268.0, 350047.0, 357296.0, 367770.0, 374650.0, 381964.0, 396972.0, 406207.0, 418693.0, 424007.0, 430473.0, 445024.0, 447972.0, 451206.0, 474337.0, 479170.0, 493960.0, 522632.0, 531686.0, 555628.0, 565852.0, 600597.0, 606792.0, 638906.0, 685403.0, 718105.0, 757719.0, 811003.0, 842224.0, 892977.0, 947706.0, 1015533.0, 1059641.0, 1110724.0, 1150042.0, 1183955.0, 1234355.0, 1277335.0, 1312408.0, 1358363.0, 1395943.0, 1416626.0, 1446726.0, 1483433.0, 1513426.0, 1552109.0, 1593886.0, 1617891.0, 1662249.0, 1695835.0, 1744514.0, 1808624.0, 1881555.0, 1916520.0, 1981056.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 2048258.0, 0.0, 63754.0, 97736.0, 136329.0, 160959.0, 243930.0, 257869.0, 285685.0, 309584.0, 331532.0, 349832.0, 367184.0, 385496.0, 414917.0, 437430.0, 456465.0, 478575.0, 490054.0, 514912.0, 525817.0, 539186.0, 546010.0, 556053.0, 582061.0, 594279.0, 600285.0, 615291.0, 624519.0, 645994.0, 693130.0, 730051.0, 745336.0, 762934.0, 795468.0, 896263.0, 933409.0, 951627.0, 996303.0, 1082489.0, 1139153.0, 1239758.0, 1322062.0, 1373785.0, 1433985.0, 1513608.0, 1578839.0, 1671364.0, 1752842.0, 1820630.0, 1914412.0, 2000536.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 0.0, 63643.0, 102552.0, 122522.0, 148833.0, 175143.0, 223216.0, 236495.0, 261020.0, 271046.0, 291262.0, 306229.0, 323128.0, 337482.0, 353926.0, 376248.0, 391467.0, 397262.0, 406439.0, 413890.0, 422561.0, 428501.0, 453401.0, 465496.0, 473129.0, 518628.0, 523736.0, 547981.0, 548111.0, 582674.0, 582674.0, 622943.0, 671785.0, 686085.0, 740518.0, 763566.0, 797747.0, 862176.0, 898533.0, 936590.0, 995422.0, 1041597.0, 1103045.0, 1167786.0, 1223520.0, 1292688.0, 1334595.0, 1415540.0, 1504041.0, 1587519.0, 1676471.0, 1744176.0, 1806850.0, 1904567.0, 1959433.0, 2029061.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 0.0, 62601.0, 117386.0, 157032.0, 192857.0, 288933.0, 306730.0, 339609.0, 358344.0, 388652.0, 399767.0, 428559.0, 443564.0, 457848.0, 470258.0, 489070.0, 506500.0, 510390.0, 530766.0, 538056.0, 546196.0, 556193.0, 565988.0, 585478.0, 591288.0, 605919.0, 612629.0, 619509.0, 657498.0, 668513.0, 684021.0, 702724.0, 732415.0, 755609.0, 780259.0, 796810.0, 829957.0, 870698.0, 925877.0, 967117.0, 1048733.0, 1113425.0, 1178802.0, 1256644.0, 1294853.0, 1350643.0, 1402619.0, 1450016.0, 1491595.0, 1559813.0, 1599419.0, 1644825.0, 1704629.0, 1727715.0, 1762355.0, 1826526.0, 1859703.0, 1875996.0, 1925568.0, 1947472.0, 1988127.0, 2003823.0, 2016921.0, 2045462.0, 2048257.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 0.0, 96097.0, 127164.0, 182783.0, 200641.0, 241275.0, 258319.0, 285263.0, 308352.0, 363804.0, 382018.0, 397598.0, 420674.0, 435206.0, 465311.0, 484153.0, 522628.0, 531106.0, 544443.0, 551590.0, 564262.0, 574810.0, 584254.0, 597033.0, 632271.0, 645993.0, 662387.0, 682286.0, 695374.0, 753476.0, 773147.0, 819052.0, 838998.0, 864574.0, 966668.0, 988490.0, 1032089.0, 1091836.0, 1171574.0, 1198197.0, 1256770.0, 1334365.0, 1398940.0, 1459628.0, 1546439.0, 1698025.0, 1786658.0, 1887423.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048257.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 0.0, 67458.0, 122185.0, 164055.0, 240849.0, 262166.0, 293417.0, 309569.0, 340852.0, 356485.0, 369428.0, 390743.0, 402314.0, 416734.0, 458645.0, 479280.0, 499303.0, 504186.0, 510127.0, 524480.0, 535381.0, 541663.0, 551940.0, 569307.0, 598265.0, 614983.0, 628308.0, 648440.0, 673837.0, 692080.0, 713467.0, 801119.0, 847224.0, 857476.0, 897851.0, 947807.0, 973134.0, 1002585.0, 1020259.0, 1062393.0, 1085755.0, 1137045.0, 1192177.0, 1233967.0, 1291755.0, 1340767.0, 1369804.0, 1455932.0, 1498982.0, 1515324.0, 1536359.0, 1589379.0, 1617321.0, 1666320.0, 1690826.0, 1716395.0, 1756348.0, 1774575.0, 1790064.0, 1847304.0, 1874468.0, 1898928.0, 1924359.0, 1997386.0, 2018350.0, 2036914.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 2048259.0, 0.0, 61496.0, 106390.0, 122633.0, 227068.0, 247362.0, 257521.0, 297202.0, 312871.0, 343188.0, 345882.0, 357501.0, 367392.0, 379902.0, 391528.0, 401712.0, 416837.0, 422718.0, 439285.0, 442389.0, 449350.0, 454021.0, 459213.0, 470796.0, 474460.0, 476796.0, 490476.0, 520533.0, 533496.0, 554277.0, 569909.0, 594629.0, 626658.0, 639256.0, 661593.0, 701021.0, 721479.0, 746039.0, 766919.0, 805340.0, 826393.0, 860840.0, 882412.0, 908857.0, 954747.0, 976476.0, 1018529.0, 1052948.0, 1075071.0, 1106389.0, 1125174.0, 1145787.0, 1207028.0, 1223060.0, 1253423.0, 1264275.0, 1320738.0, 1331269.0, 1410161.0, 1435873.0, 1464260.0, 1479798.0, 1503753.0, 1564143.0, 1594453.0, 1610961.0, 1643414.0, 1648831.0, 1734073.0, 1751896.0, 1764975.0, 1784694.0, 1789530.0, 1812757.0, 1855998.0, 1899265.0, 1945192.0, 1955925.0, 1982221.0, 2045480.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0, 2048260.0]
num = [0.0, 63754.0, 97736.0, 136329.0, 160959.0, 243930.0, 257869.0, 285685.0, 309584.0, 331532.0, 349832.0, 367184.0, 385496.0, 414917.0, 437430.0, 456465.0, 478575.0, 490054.0, 514912.0, 525817.0, 539186.0, 546010.0, 556053.0, 582061.0, 594279.0, 600285.0, 615291.0, 624519.0, 645994.0, 693130.0, 730051.0, 745336.0, 762934.0, 795468.0, 896263.0, 933409.0, 951627.0, 996303.0, 1082489.0, 1139153.0, 1239758.0, 1322062.0, 1378626.0, 1443628.0, 1530259.0, 1599813.0, 1699177.0, 1785298.0, 1856329.0, 1955868.0, 2047789.0, 2115178.0, 2219797.0, 2296560.0, 2389563.0, 2456706.0, 2547826.0, 2636169.0, 2720750.0, 2791010.0, 2880613.0, 2964392.0, 3039166.0, 3120093.0, 3205109.0, 3285884.0, 3344528.0, 3440147.0, 3540367.0, 3628561.0, 3696065.0, 3784565.0, 3900981.0, 3961596.0, 4026716.0, 4097403.0, 4179393.0, 4275930.0, 4363520.0, 4476027.0, 4567007.0, 4682330.0, 4743348.0, 4889754.0, 4985139.0, 5099184.0, 5209347.0, 5267160.0, 5397621.0, 5484026.0, 5600212.0, 5724758.0, 5825581.0, 5960804.0]
A = [i for i in range(len(num))]


plt.plot(A, num)
plt.title('Cumulative budget statistics')
plt.xlabel('time')
plt.ylabel('used budget')
plt.show()
