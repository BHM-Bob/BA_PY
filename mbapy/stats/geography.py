#from https://www.php1.cn/detail/Python_BanZhongG_7c9b41d0.html

provinces = {
  '吉林省': [125.326800, 43.896160], '黑龙江省': [126.662850, 45.742080],
  '辽宁省': [123.429250, 41.835710], '内蒙古自治区': [111.765220, 40.817330],
  '新疆维吾尔自治区': [87.627100, 43.793430], '青海省': [101.780110, 36.620870],
  '北京市': [116.407170, 39.904690], '天津市': [117.199370, 39.085100],
  '上海市': [121.473700, 31.230370], '重庆市': [106.550730, 29.564710],
  '河北省': [114.469790, 38.035990], '河南省': [113.753220, 34.765710],
  '陕西省': [108.954240, 34.264860], '江苏省': [118.762950, 32.060710],
  '山东省': [117.020760, 36.668260], '山西省': [112.562720, 37.873430],
  '甘肃省': [103.826340, 36.059420], '宁夏回族自治区': [106.258670, 38.471170],
  '四川省': [104.075720, 30.650890], '西藏自治区': [91.117480, 29.647250],
  '安徽省': [117.285650, 31.861570], '浙江省': [120.153600, 30.265550],
  '湖北省': [114.342340, 30.545390], '湖南省': [112.983400, 28.112660],
  '福建省': [119.296590, 26.099820], '江西省': [115.910040, 28.674170],
  '贵州省': [106.707220, 26.598200], '云南省': [102.709730, 25.045300],
  '广东省': [113.266270, 23.131710], '广西壮族自治区': [108.327540, 22.815210],
  '香港': [114.165460, 22.275340], '澳门': [113.549130, 22.198750],
  '海南省': [110.348630, 20.019970], '台湾省': [121.520076, 25.030724],
}

city = [
["北京市","",116.407170,39.904690],
["天津市","",117.199370,39.085100],
["上海市","",121.473700,31.230370],
["重庆市","",106.550730,29.564710],
["香港","",114.165460,22.275340],
["澳门","",113.549130,22.198750],
["河北省","石家庄市",114.514300,38.042760],
["河北省","唐山市",118.180580,39.630480],
["河北省","秦皇岛市",119.599640,39.935450],
["河北省","邯郸市",114.539180,36.625560],
["河北省","邢台市",114.504430,37.070550],
["河北省","保定市",115.464590,38.873960],
["河北省","张家口市",114.887550,40.824440],
["河北省","承德市",117.963400,40.951500],
["河北省","沧州市",116.838690,38.304410],
["河北省","廊坊市",116.683760,39.537750],
["河北省","衡水市",115.670540,37.738860],
["河南省","郑州市",113.624930,34.747250],
["河南省","开封市",114.307310,34.797260],
["河南省","洛阳市",112.453610,34.618120],
["河南省","平顶山市",113.192410,33.766090],
["河南省","安阳市",114.393100,36.097710],
["河南省","鹤壁市",114.297450,35.747000],
["河南省","新乡市",113.926750,35.303230],
["河南省","焦作市",113.242010,35.215630],
["河南省","濮阳市",115.029320,35.761890],
["河南省","许昌市",113.852330,34.035700],
["河南省","漯河市",114.016810,33.581490],
["河南省","三门峡市",111.200300,34.772610],
["河南省","南阳市",112.528510,32.990730],
["河南省","商丘市",115.656350,34.414270],
["河南省","信阳市",114.092790,32.147140],
["河南省","周口市",114.696950,33.625830],
["河南省","驻马店市",114.022990,33.011420],
["山东省","济南市",117.120090,36.651840],
["山东省","青岛市",120.382990,36.066230],
["山东省","淄博市",118.054800,36.813100],
["山东省","枣庄市",117.321960,34.810710],
["山东省","东营市",118.674660,37.433650],
["山东省","烟台市",121.448010,37.463530],
["山东省","潍坊市",119.161760,36.706860],
["山东省","济宁市",116.587240,35.414590],
["山东省","泰安市",117.088400,36.199940],
["山东省","威海市",122.121710,37.513480],
["山东省","日照市",119.527190,35.416460],
["山东省","莱芜市",117.676670,36.213590],
["山东省","临沂市",118.356460,35.104650],
["山东省","德州市",116.359270,37.435500],
["山东省","聊城市",115.985490,36.457020],
["山东省","滨州市",117.972790,37.382110],
["山东省","菏泽市",115.481150,35.233630],
["山西省","太原市",112.556252,37.876876],
["山西省","大同市",113.304424,40.081863],
["山西省","阳泉市",113.580470,37.856680],
["山西省","长治市",113.116490,36.195810],
["山西省","晋城市",112.851130,35.490390],
["山西省","朔州市",112.439374,39.357422],
["山西省","晋中市",112.752780,37.687020],
["山西省","运城市",111.006990,35.026280],
["山西省","忻州市",112.734180,38.416700],
["山西省","临汾市",111.519620,36.088220],
["山西省","吕梁市",111.141650,37.519340],
["辽宁省","沈阳市",123.463100,41.677180],
["辽宁省","大连市",121.614760,38.913690],
["辽宁省","鞍山市",122.994600,41.107770],
["辽宁省","抚顺市",123.957220,41.879710],
["辽宁省","本溪市",123.766860,41.294130],
["辽宁省","丹东市",124.356010,39.999800],
["辽宁省","锦州市",121.127030,41.095150],
["辽宁省","营口市",122.234900,40.666830],
["辽宁省","阜新市",121.670110,42.021660],
["辽宁省","辽阳市",123.237360,41.268090],
["辽宁省","盘锦市",122.070780,41.119960],
["辽宁省","铁岭市",123.842410,42.286200],
["辽宁省","朝阳市",120.450800,41.573470],
["辽宁省","葫芦岛市",120.836990,40.711000],
["吉林省","长春市",125.323570,43.816020],
["吉林省","吉林市",126.549440,43.837840],
["吉林省","四平市",124.350360,43.166460],
["吉林省","辽源市",125.143680,42.888050],
["吉林省","通化市",125.939900,41.728290],
["吉林省","白山市",126.424430,41.940800],
["吉林省","松原市",124.825150,45.141100],
["吉林省","白城市",122.838710,45.619600],
["吉林省","延边朝鲜族自治州",129.509100,42.891190],
["黑龙江省","哈尔滨市",126.535800,45.802160],
["黑龙江省","齐齐哈尔市",123.917960,47.354310],
["黑龙江省","鸡西市",130.969540,45.295240],
["黑龙江省","鹤岗市",130.297850,47.349890],
["黑龙江省","双鸭山市",131.159100,46.646580],
["黑龙江省","大庆市",125.110961,46.595319],
["黑龙江省","伊春市",128.840490,47.727520],
["黑龙江省","佳木斯市",130.318820,46.799770],
["黑龙江省","七台河市",131.003060,45.770650],
["黑龙江省","牡丹江市",129.632440,44.552690],
["黑龙江省","黑河市",127.528520,50.245230],
["黑龙江省","绥化市",126.969320,46.652460],
["黑龙江省","大兴安岭地区",124.592160,51.923980],
["江苏省","南京市",118.796470,32.058380],
["江苏省","无锡市",120.312370,31.490990],
["江苏省","徐州市",117.285770,34.204400],
["江苏省","常州市",119.973650,31.810720],
["江苏省","苏州市",120.583190,31.298340],
["江苏省","南通市",120.893710,31.979580],
["江苏省","连云港市",119.222950,34.596690],
["江苏省","淮安市",119.015950,33.610160],
["江苏省","盐城市",120.161640,33.349510],
["江苏省","扬州市",119.412690,32.393580],
["江苏省","镇江市",119.425000,32.189590],
["江苏省","泰州市",119.925540,32.455460],
["江苏省","宿迁市",118.275490,33.961930],
["浙江省","杭州市",120.155150,30.274150],
["浙江省","宁波市",121.550270,29.873860],
["浙江省","温州市",120.699390,27.994920],
["浙江省","嘉兴市",120.755500,30.745010],
["浙江省","湖州市",120.088050,30.893050],
["浙江省","绍兴市",120.580200,30.030330],
["浙江省","金华市",119.647590,29.078120],
["浙江省","衢州市",118.874190,28.935920],
["浙江省","舟山市",122.207780,29.985390],
["浙江省","台州市",121.420560,28.656110],
["浙江省","丽水市",119.922930,28.467200],
["安徽省","合肥市",117.229010,31.820570],
["安徽省","芜湖市",118.433130,31.352460],
["安徽省","蚌埠市",117.389320,32.915480],
["安徽省","淮南市",116.999800,32.625490],
["安徽省","马鞍山市",118.506110,31.670670],
["安徽省","淮北市",116.798340,33.954790],
["安徽省","铜陵市",117.812320,30.944860],
["安徽省","安庆市",117.063540,30.542940],
["安徽省","黄山市",118.338660,29.715170],
["安徽省","滁州市",118.316830,32.301810],
["安徽省","阜阳市",115.814950,32.889630],
["安徽省","宿州市",116.963910,33.646140],
["安徽省","六安市",116.523240,31.734880],
["安徽省","亳州市",115.779310,33.844610],
["安徽省","池州市",117.491420,30.664690],
["安徽省","宣城市",118.758660,30.940780],
["福建省","福州市",119.296470,26.074210],
["福建省","厦门市",118.089480,24.479510],
["福建省","莆田市",119.007710,25.454000],
["福建省","三明市",117.639220,26.263850],
["福建省","泉州市",118.675870,24.873890],
["福建省","漳州市",117.647250,24.513470],
["福建省","南平市",118.120430,27.331750],
["福建省","龙岩市",117.017220,25.075040],
["福建省","宁德市",119.548190,26.665710],
["江西省","南昌市",115.857940,28.682020],
["江西省","景德镇市",117.178390,29.268690],
["江西省","萍乡市",113.854270,27.622890],
["江西省","九江市",116.001460,29.705480],
["江西省","新余市",114.917130,27.817760],
["江西省","鹰潭市",117.069190,28.260190],
["江西省","赣州市",114.934760,25.831090],
["江西省","吉安市",114.993760,27.113820],
["江西省","宜春市",114.416120,27.814430],
["江西省","抚州市",116.358090,27.947810],
["江西省","上饶市",117.943570,28.454630],
["湖北省","武汉市",114.305250,30.592760],
["湖北省","黄石市",115.038900,30.199530],
["湖北省","十堰市",110.798010,32.629180],
["湖北省","宜昌市",111.286420,30.691860],
["湖北省","襄阳市",112.122550,32.009000],
["湖北省","鄂州市",114.894950,30.390850],
["湖北省","荆门市",112.199450,31.035460],
["湖北省","孝感市",113.916450,30.924830],
["湖北省","荆州市",112.240690,30.334790],
["湖北省","黄冈市",114.872380,30.453470],
["湖北省","咸宁市",114.322450,29.841260],
["湖北省","随州市",113.382620,31.690130],
["湖北省","恩施土家族苗族自治州",109.488170,30.272170],
["湖南省","长沙市",112.938860,28.227780],
["湖南省","株洲市",113.133960,27.827670],
["湖南省","湘潭市",112.944110,27.829750],
["湖南省","衡阳市",112.571950,26.893240],
["湖南省","邵阳市",111.467700,27.238900],
["湖南省","岳阳市",113.129190,29.357280],
["湖南省","常德市",111.698540,29.031580],
["湖南省","张家界市",110.478390,29.116670],
["湖南省","益阳市",112.355160,28.553910],
["湖南省","郴州市",113.014850,25.770630],
["湖南省","永州市",111.612250,26.420340],
["湖南省","怀化市",110.001600,27.569740],
["湖南省","娄底市",111.994580,27.697280],
["湖南省","湘西土家族苗族自治州",109.738930,28.311730],
["广东省","广州市",113.264360,23.129080],
["广东省","韶关市",113.597230,24.810390],
["广东省","深圳市",114.059560,22.542860],
["广东省","珠海市",113.576680,22.270730],
["广东省","汕头市",116.682210,23.353500],
["广东省","佛山市",113.121920,23.021850],
["广东省","江门市",113.081610,22.578650],
["广东省","湛江市",110.358940,21.271340],
["广东省","茂名市",110.925230,21.663290],
["广东省","肇庆市",112.465280,23.046900],
["广东省","惠州市",114.416790,23.110750],
["广东省","梅州市",116.122640,24.288440],
["广东省","汕尾市",115.375140,22.785660],
["广东省","河源市",114.700650,23.743650],
["广东省","阳江市",111.982560,21.858290],
["广东省","清远市",113.056150,23.682010],
["广东省","东莞市",113.751790,23.020670],
["广东省","中山市",113.392600,22.515950],
["广东省","潮州市",116.622960,23.656700],
["广东省","揭阳市",116.372710,23.549720],
["广东省","云浮市",112.044530,22.915250],
["海南省","海口市",110.199890,20.044220],
["海南省","三亚市",109.512090,18.252480],
["海南省","三沙市",112.333560,16.832720],
["海南省","儋州市",109.580690,19.520930],
["四川省","成都市",104.064760,30.570200],
["四川省","自贡市",104.778440,29.339200],
["四川省","攀枝花市",101.718720,26.582280],
["四川省","泸州市",105.442570,28.871700],
["四川省","德阳市",104.397900,31.126790],
["四川省","绵阳市",104.679600,31.467510],
["四川省","广元市",105.843570,32.435490],
["四川省","遂宁市",105.592730,30.532860],
["四川省","内江市",105.058440,29.580150],
["四川省","乐山市",103.765390,29.552210],
["四川省","南充市",106.110730,30.837310],
["四川省","眉山市",103.848510,30.075630],
["四川省","宜宾市",104.641700,28.751300],
["四川省","广安市",106.633220,30.455960],
["四川省","达州市",107.467910,31.208640],
["四川省","雅安市",103.042400,30.010530],
["四川省","巴中市",106.747330,31.867150],
["四川省","资阳市",104.627980,30.128590],
["四川省","阿坝藏族羌族自治州",102.224770,31.899400],
["四川省","甘孜藏族自治州",101.962540,30.049320],
["四川省","凉山彝族自治州",102.267460,27.881640],
["贵州省","贵阳市",106.630240,26.647020],
["贵州省","六盘水市",104.830230,26.593360],
["贵州省","遵义市",106.927230,27.725450],
["贵州省","安顺市",105.946200,26.253670],
["贵州省","毕节市",105.305040,27.298470],
["贵州省","铜仁市",109.180990,27.690660],
["贵州省","黔西南布依族苗族自治州",104.904370,25.089880],
["贵州省","黔东南苗族侗族自治州",107.984160,26.583640],
["贵州省","黔南布依族苗族自治州",107.522260,26.254270],
["云南省","昆明市",102.833220,24.879660],
["云南省","曲靖市",103.796250,25.490020],
["云南省","玉溪市",102.547140,24.351800],
["云南省","保山市",99.161810,25.112050],
["云南省","昭通市",103.716800,27.338170],
["云南省","丽江市",100.227100,26.856480],
["云南省","普洱市",100.966240,22.825210],
["云南省","临沧市",100.088840,23.884260],
["云南省","楚雄彝族自治州",101.527670,25.044950],
["云南省","红河哈尼族彝族自治州",103.375600,23.364220],
["云南省","文山壮族苗族自治州",104.215040,23.398490],
["云南省","西双版纳傣族自治州",100.797390,22.007490],
["云南省","大理白族自治州",100.267640,25.606480],
["云南省","德宏傣族景颇族自治州",98.584860,24.432320],
["云南省","怒江傈僳族自治州",98.856700,25.817630],
["云南省","迪庆藏族自治州",99.703050,27.819080],
["陕西省","西安市",108.939840,34.341270],
["陕西省","铜川市",108.945150,34.896730],
["陕西省","宝鸡市",107.237320,34.361940],
["陕西省","咸阳市",108.709290,34.329320],
["陕西省","渭南市",109.510150,34.499970],
["陕西省","延安市",109.489780,36.585290],
["陕西省","汉中市",107.023770,33.067610],
["陕西省","榆林市",109.734580,38.285200],
["陕西省","安康市",109.029320,32.684860],
["陕西省","商洛市",109.940410,33.870360],
["甘肃省","兰州市",103.834170,36.061380],
["甘肃省","嘉峪关市",98.290110,39.772010],
["甘肃省","金昌市",102.187590,38.520060],
["甘肃省","白银市",104.137730,36.544700],
["甘肃省","天水市",105.724860,34.580850],
["甘肃省","武威市",102.637970,37.928200],
["甘肃省","张掖市",100.449810,38.925920],
["甘肃省","平凉市",106.665300,35.543030],
["甘肃省","酒泉市",98.493940,39.732550],
["甘肃省","庆阳市",107.642920,35.709780],
["甘肃省","定西市",104.625240,35.581130],
["甘肃省","陇南市",104.921660,33.401000],
["甘肃省","临夏回族自治州",103.210910,35.601220],
["甘肃省","甘南藏族自治州",102.911020,34.983270],
["青海省","西宁市",101.777820,36.617290],
["青海省","海东市",102.401730,36.482090],
["青海省","海北藏族自治州",100.900960,36.954540],
["青海省","黄南藏族自治州",102.015070,35.519910],
["青海省","海南藏族自治州",100.620370,36.286630],
["青海省","果洛藏族自治州",100.244750,34.471410],
["青海省","玉树藏族自治州",97.006500,33.005280],
["青海省","海西蒙古族藏族自治州",97.371220,37.377100],
["广西壮族自治区","南宁市",108.366900,22.816730],
["广西壮族自治区","柳州市",109.415520,24.325430],
["广西壮族自治区","桂林市",110.290020,25.273610],
["广西壮族自治区","梧州市",111.279170,23.476910],
["广西壮族自治区","北海市",109.120080,21.481120],
["广西壮族自治区","防城港市",108.354720,21.687130],
["广西壮族自治区","钦州市",108.654310,21.979700],
["广西壮族自治区","贵港市",109.597640,23.113060],
["广西壮族自治区","玉林市",110.180980,22.654510],
["广西壮族自治区","百色市",106.618380,23.902160],
["广西壮族自治区","贺州市",111.566550,24.403460],
["广西壮族自治区","河池市",108.085400,24.692910],
["广西壮族自治区","来宾市",109.222380,23.752100],
["广西壮族自治区","崇左市",107.364850,22.378950],
["内蒙古自治区","呼和浩特市",111.751990,40.841490],
["内蒙古自治区","包头市",109.840210,40.657810],
["内蒙古自治区","乌海市",106.795460,39.653840],
["内蒙古自治区","赤峰市",118.888940,42.258600],
["内蒙古自治区","通辽市",122.244690,43.652470],
["内蒙古自治区","鄂尔多斯市",109.780870,39.608450],
["内蒙古自治区","呼伦贝尔市",119.765840,49.211630],
["内蒙古自治区","巴彦淖尔市",107.387730,40.743170],
["内蒙古自治区","乌兰察布市",113.133760,40.993910],
["内蒙古自治区","兴安盟",122.038180,46.082080],
["内蒙古自治区","锡林郭勒盟",116.047750,43.933200],
["内蒙古自治区","阿拉善盟",105.728980,38.851530],
["宁夏回族自治区","银川市",106.232480,38.486440],
["宁夏回族自治区","石嘴山市",106.384180,38.984100],
["宁夏回族自治区","吴忠市",106.198790,37.997550],
["宁夏回族自治区","固原市",106.242590,36.015800],
["宁夏回族自治区","中卫市",105.196760,37.500260],
["西藏自治区","拉萨市",91.114500,29.644150],
["西藏自治区","日喀则市",88.881160,29.267050],
["西藏自治区","昌都市",97.172250,31.140730],
["西藏自治区","林芝市",94.361550,29.648950],
["西藏自治区","山南市",91.773130,29.237050],
["西藏自治区","那曲市",92.051360,31.476140],
["西藏自治区","阿里地区",81.145400,30.400510],
["新疆维吾尔自治区","乌鲁木齐市",87.616880,43.826630],
["新疆维吾尔自治区","克拉玛依市",84.889270,45.579990],
["新疆维吾尔自治区","吐鲁番市",89.189540,42.951300],
["新疆维吾尔自治区","哈密市",93.515380,42.818550],
["新疆维吾尔自治区","昌吉回族自治州",87.308220,44.011170],
["新疆维吾尔自治区","博尔塔拉蒙古自治州",82.066650,44.905970],
["新疆维吾尔自治区","巴音郭楞蒙古自治州",86.145170,41.764040],
["新疆维吾尔自治区","阿克苏地区",80.260080,41.168420],
["新疆维吾尔自治区","克孜勒苏柯尔克孜自治州",76.166610,39.715300],
["新疆维吾尔自治区","喀什地区",75.989760,39.470420],
["新疆维吾尔自治区","和田地区",79.922470,37.114310],
["新疆维吾尔自治区","伊犁哈萨克自治州",81.324160,43.916890],
["新疆维吾尔自治区","塔城地区",82.980460,46.745320],
["新疆维吾尔自治区","阿勒泰地区",88.140230,47.845640],
]