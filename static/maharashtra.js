const CANDIDATES = [
  {id:1,name:"Amol Haribhau Jawale",party:"BJP",constituency:"RAVER",criminal:0,education:"Post Graduate",assets:""},
  {id:2,name:"Abdul Sattar",party:"Shiv Sena",constituency:"SILLOD",criminal:0,education:"Graduate",assets:""},
  {id:3,name:"Abhijeet Dhananjay Patil",party:"NCP-Sharadchandra Pawar",constituency:"MADHA",criminal:0,education:"10th Pass",assets:""},
  {id:4,name:"Abhimanyu Dattatray Pawar",party:"BJP",constituency:"AUSA",criminal:0,education:"Post Graduate",assets:""},
  {id:5,name:"Abitkar Prakash Anandrao",party:"Shiv Sena",constituency:"RADHANAGARI",criminal:0,education:"Graduate",assets:""},
  {id:6,name:"Abu Asim Azmi",party:"SP",constituency:"MANKHURD SHIVAJINAGAR",criminal:0,education:"Graduate",assets:""},
  {id:7,name:"Aditi Sunil Tatkare",party:"NCP",constituency:"SHRIVARDHAN",criminal:0,education:"Post Graduate",assets:""},
  {id:8,name:"Aditya Uddhav Thackeray",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"WORLI",criminal:0,education:"Graduate Professional",assets:""},
  {id:9,name:"Adv Rahul Suresh Narwekar",party:"BJP",constituency:"COLABA",criminal:0,education:"Graduate Professional",assets:""},
  {id:10,name:"Adv. Ashish Shelar",party:"BJP",constituency:"VANDRE WEST",criminal:0,education:"Graduate Professional",assets:""},
  {id:11,name:"Adv. Kokate Manikrao Shivaji",party:"NCP",constituency:"SINNAR",criminal:0,education:"Graduate Professional",assets:""},
  {id:12,name:"Adv.Rahul Uttamrao Dhikale",party:"BJP",constituency:"NASHIK EAST",criminal:0,education:"Graduate Professional",assets:""},
  {id:13,name:"Agrawal Anupbhaiyya Omprakash",party:"BJP",constituency:"DHULE CITY",criminal:0,education:"12th Pass",assets:""},
  {id:14,name:"Agrawal Vinod",party:"BJP",constituency:"GONDIA",criminal:0,education:"12th Pass",assets:""},
  {id:15,name:"Ahire Saroj Babulal",party:"NCP",constituency:"DEOLALI (SC)",criminal:0,education:"Graduate",assets:""},
  {id:16,name:"Ailani Kumar Uttamchand",party:"BJP",constituency:"ULHASNAGAR",criminal:0,education:"10th Pass",assets:""},
  {id:17,name:"Ajit Anantrao Pawar",party:"NCP",constituency:"BARAMATI",criminal:0,education:"12th Pass",assets:""},
  {id:18,name:"Akash Pandurang Fundkar",party:"BJP",constituency:"KHAMGAON",criminal:0,education:"Graduate Professional",assets:""},
  {id:19,name:"Alavani Parag",party:"BJP",constituency:"VILE PARLE",criminal:0,education:"Graduate Professional",assets:""},
  {id:20,name:"Amal Mahadevrao Mahadik",party:"BJP",constituency:"KOLHAPUR SOUTH",criminal:0,education:"Graduate",assets:""},
  {id:21,name:"Ameet Bhaskar Satam",party:"BJP",constituency:"ANDHERI WEST",criminal:0,education:"Post Graduate",assets:""},
  {id:22,name:"Ameet Subhashrao Zanak",party:"INC",constituency:"RISOD",criminal:0,education:"Graduate Professional",assets:""},
  {id:23,name:"Amin Patel",party:"INC",constituency:"MUMBADEVI",criminal:0,education:"10th Pass",assets:""},
  {id:24,name:"Amit Vilasrao Deshmukh",party:"INC",constituency:"LATUR CITY",criminal:0,education:"Graduate Professional",assets:""},
  {id:25,name:"Amol Dhondiba Khatal",party:"Shiv Sena",constituency:"SANGMNER",criminal:0,education:"Graduate",assets:""},
  {id:26,name:"Amshya Fulji Padvi",party:"Shiv Sena",constituency:"AKKALKUWA (ST)",criminal:0,education:"12th Pass",assets:""},
  {id:27,name:"Anand Shankar Tidke",party:"Shiv Sena",constituency:"NANDED SOUTH",criminal:0,education:"12th Pass",assets:""},
  {id:28,name:"Anant (Bala) B. Nar",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"JOGESHWARI EAST",criminal:0,education:"10th Pass",assets:""},
  {id:29,name:"Anil Alias Balasaheb Shankarrao Mangulkar",party:"INC",constituency:"YAVATMAL",criminal:0,education:"12th Pass",assets:""},
  {id:30,name:"Anil Bhaidas Patil",party:"NCP",constituency:"AMALNER",criminal:0,education:"Post Graduate",assets:""},
  {id:31,name:"Anna Dadu Bansode",party:"NCP",constituency:"PIMPRI (SC)",criminal:0,education:"Others",assets:""},
  {id:32,name:"Antapurkar Jitesh Raosaheb",party:"BJP",constituency:"DEGLUR (SC)",criminal:0,education:"Graduate Professional",assets:""},
  {id:33,name:"Arjun Panditrao Khotkar",party:"Shiv Sena",constituency:"JALNA",criminal:0,education:"Graduate",assets:""},
  {id:34,name:"Ashish Nandkishore Jaiswal(Vakil)",party:"Shiv Sena",constituency:"RAMTEK",criminal:0,education:"Graduate Professional",assets:""},
  {id:35,name:"Ashok Dharmaraj Patil",party:"Shiv Sena",constituency:"BHANDUP WEST",criminal:0,education:"Graduate Professional",assets:""},
  {id:36,name:"Ashutosh Ashokrao Kale",party:"NCP",constituency:"KOPARGAON",criminal:0,education:"Post Graduate",assets:""},
  {id:37,name:"Aslam Ramjanali Shaikh",party:"INC",constituency:"MALAD WEST",criminal:0,education:"8th Pass",assets:""},
  {id:38,name:"Atram Dharamraobaba Bhagwantrao",party:"NCP",constituency:"AHERI (ST)",criminal:0,education:"10th Pass",assets:""},
  {id:39,name:"Atul Bhatkhalkar",party:"BJP",constituency:"KANDIVALI EAST",criminal:0,education:"Graduate",assets:""},
  {id:40,name:"Atul Moreshwar Save",party:"BJP",constituency:"AURANGABAD EAST",criminal:0,education:"Graduate",assets:""},
  {id:41,name:"Babaji Ramchandra Kale",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"KHED ALANDI",criminal:0,education:"Graduate Professional",assets:""},
  {id:42,name:"Babanrao Dattatray Yadav (Lonikar)",party:"BJP",constituency:"PARTUR",criminal:0,education:"5th Pass",assets:""},
  {id:43,name:"Babar Suhas Anilbhau",party:"Shiv Sena",constituency:"KHANPUR",criminal:0,education:"Post Graduate",assets:""},
  {id:44,name:"Babasaheb Mohanrao Patil",party:"NCP",constituency:"AHMEDPUR",criminal:0,education:"Graduate",assets:""},
  {id:45,name:"Badole Rajkumar Sudam",party:"NCP",constituency:"ARJUNI MORGAON (SC)",criminal:0,education:"Post Graduate",assets:""},
  {id:46,name:"Balaji Devidasrao Kalyankar",party:"Shiv Sena",constituency:"NANDED NORTH",criminal:0,education:"12th Pass",assets:""},
  {id:47,name:"Bangar Santosh Laxmanrao",party:"Shiv Sena",constituency:"KALAMNURI",criminal:0,education:"5th Pass",assets:""},
  {id:48,name:"Bankar Diliprao Shankarrao",party:"NCP",constituency:"NIPHAD",criminal:0,education:"Graduate",assets:""},
  {id:49,name:"Bapusaheb Tukaram Pathare",party:"NCP-Sharadchandra Pawar",constituency:"VADGAON SHERI",criminal:0,education:"8th Pass",assets:""},
  {id:50,name:"Bhimrao Dhondiba Tapkir",party:"BJP",constituency:"KHADAKWASALA",criminal:0,education:"12th Pass",assets:""},
  {id:51,name:"Bhimrao Ramji Keram",party:"BJP",constituency:"KINWAT",criminal:0,education:"10th Pass",assets:""},
  {id:52,name:"Bhondekar Narendra Bhojraj",party:"Shiv Sena",constituency:"BHANDARA (SC)",criminal:0,education:"Post Graduate",assets:""},
  {id:53,name:"Bhoye Harishchandra Sakharam",party:"BJP",constituency:"VIKRAMGAD (ST)",criminal:0,education:"10th Pass",assets:""},
  {id:54,name:"Bhumre Vilas Sandipanrao",party:"Shiv Sena",constituency:"PAITHAN",criminal:0,education:"Post Graduate",assets:""},
  {id:55,name:"Bordikar Meghna Deepak Sakore",party:"BJP",constituency:"JINTUR",criminal:0,education:"Post Graduate",assets:""},
  {id:56,name:"Bornare (Sir) Ramesh Nanasaheb",party:"Shiv Sena",constituency:"VAIJAPUR",criminal:0,education:"Post Graduate",assets:""},
  {id:57,name:"Captain R Tamil Selvan",party:"BJP",constituency:"SION KOLIWADA",criminal:0,education:"10th Pass",assets:""},
  {id:58,name:"Chainsukh Madanlal Sancheti",party:"BJP",constituency:"MALKAPUR",criminal:0,education:"Graduate",assets:""},
  {id:59,name:"Chandradeep Shashikant Narake",party:"Shiv Sena",constituency:"KARVIR",criminal:0,education:"Graduate Professional",assets:""},
  {id:60,name:"Chandrakant (Dada) Bachhu Patil",party:"BJP",constituency:"KOTHRUD",criminal:0,education:"Graduate",assets:""},
  {id:61,name:"Chandrakant Alias Rajubhaiya Ramakant Navghare",party:"NCP",constituency:"BASMATH",criminal:0,education:"Post Graduate",assets:""},
  {id:62,name:"Chandrakant Baliram Sonawane",party:"Shiv Sena",constituency:"CHOPDA (ST)",criminal:0,education:"Post Graduate",assets:""},
  {id:63,name:"Chandrakant Nimba Patil",party:"Shiv Sena",constituency:"MUKTAINAGAR",criminal:0,education:"Graduate",assets:""},
  {id:64,name:"Chandrashekhar Krushnarao Bawankule",party:"BJP",constituency:"KAMTHI",criminal:0,education:"12th Pass",assets:""},
  {id:65,name:"Chaudhary Manisha Ashok",party:"BJP",constituency:"DAHISAR",criminal:0,education:"Graduate",assets:""},
  {id:66,name:"Chavan Ravindra Dattatray",party:"BJP",constituency:"DOMBIVALI",criminal:0,education:"Graduate",assets:""},
  {id:67,name:"Chavan Sreejaya Ashokrao",party:"BJP",constituency:"BHOKAR",criminal:0,education:"Graduate Professional",assets:""},
  {id:68,name:"Chetan Vitthal Tupe",party:"NCP",constituency:"HADAPSAR",criminal:0,education:"10th Pass",assets:""},
  {id:69,name:"Chhagan Bhujbal",party:"NCP",constituency:"YEVLA",criminal:0,education:"10th Pass",assets:""},
  {id:70,name:"Choughule Mahesh Prabhakar",party:"BJP",constituency:"BHIWANDI WEST",criminal:0,education:"10th Pass",assets:""},
  {id:71,name:"Dadaji Dagdu Bhuse",party:"Shiv Sena",constituency:"MALEGAON OUTER",criminal:0,education:"Others",assets:""},
  {id:72,name:"Dalitmitra Dr.Ashokrao Mane(Bapu",party:"Jan Surajya Shakti",constituency:"HATKANANGLE (SC)",criminal:0,education:"10th Pass",assets:""},
  {id:73,name:"Datke Pravin Prabhakarrao",party:"BJP",constituency:"NAGPUR CENTRAL",criminal:0,education:"12th Pass",assets:""},
  {id:74,name:"Dattatraya Vithoba Bharane",party:"NCP",constituency:"INDAPUR",criminal:0,education:"12th Pass",assets:""},
  {id:75,name:"Daulat Bhika Daroda",party:"NCP",constituency:"SHAHAPUR (ST)",criminal:0,education:"Graduate",assets:""},
  {id:76,name:"Deepak Vasantrao Kesarkar",party:"Shiv Sena",constituency:"SAWANTWADI",criminal:0,education:"Graduate",assets:""},
  {id:77,name:"Deorao Vithoba Bhongle",party:"BJP",constituency:"RAJURA",criminal:0,education:"Post Graduate",assets:""},
  {id:78,name:"Derkar Sanjay Nilkanthrao",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"WANI",criminal:0,education:"Post Graduate",assets:""},
  {id:79,name:"Desai Shambhuraj Shivajirao",party:"Shiv Sena",constituency:"PATAN",criminal:0,education:"12th Pass",assets:""},
  {id:80,name:"Deshmukh Satyajit Shivajirao",party:"BJP",constituency:"SHIRALA",criminal:0,education:"Graduate",assets:""},
  {id:81,name:"Deshmukh Vijay Sidramappa",party:"BJP",constituency:"SOLAPUR CITY NORTH",criminal:0,education:"Graduate",assets:""},
  {id:82,name:"Devendra Gangadhar Fadnavis",party:"BJP",constituency:"NAGPUR SOUTH WEST",criminal:0,education:"Graduate Professional",assets:""},
  {id:83,name:"Devendra Rajesh Kothe",party:"BJP",constituency:"SOLAPUR CITY CENTRAL",criminal:0,education:"Graduate Professional",assets:""},
  {id:84,name:"Devyani Suhas Pharande",party:"BJP",constituency:"NASHIK (CENTRAL)",criminal:0,education:"Post Graduate",assets:""},
  {id:85,name:"Dhananjay Hari Gadgil",party:"BJP",constituency:"SANGLI",criminal:0,education:"Graduate",assets:""},
  {id:86,name:"Dhananjay Panditrao Munde",party:"NCP",constituency:"PARLI",criminal:0,education:"Graduate Professional",assets:""},
  {id:87,name:"Dhas Suresh Ramchandra",party:"BJP",constituency:"ASHTI",criminal:0,education:"Graduate",assets:""},
  {id:88,name:"Dilip Bhausaheb Lande",party:"Shiv Sena",constituency:"CHANDIVALI",criminal:0,education:"Graduate",assets:""},
  {id:89,name:"Dilip Gangadhar Sopal",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"BARSHI",criminal:0,education:"Graduate Professional",assets:""},
  {id:90,name:"Dilip Manglu Borse",party:"BJP",constituency:"BAGLAN (ST)",criminal:0,education:"Graduate",assets:""},
  {id:91,name:"Dnyaneshwar Alias Mauli Aba Katke",party:"NCP",constituency:"SHIRUR",criminal:0,education:"12th Pass",assets:""},
  {id:92,name:"Dr Ashishrao Deshmukh",party:"BJP",constituency:"SAVNER",criminal:0,education:"Doctorate",assets:""},
  {id:93,name:"Dr Atulbaba Suresh Bhosale",party:"BJP",constituency:"KARAD SOUTH",criminal:0,education:"Graduate Professional",assets:""},
  {id:94,name:"Dr Nitin Kashinath Raut",party:"INC",constituency:"NAGPUR NORTH (SC)",criminal:0,education:"Doctorate",assets:""},
  {id:95,name:"Dr. Babasaheb Annasaheb Deshmukh",party:"Peasants And Workers Party of India",constituency:"SANGOLA",criminal:0,education:"Post Graduate",assets:""},
  {id:96,name:"Dr. Balaji Pralhad Kinikar",party:"Shiv Sena",constituency:"AMBARNATH (SC)",criminal:0,education:"Graduate Professional",assets:""},
  {id:97,name:"Dr. Kiran Yamaji Lahamate",party:"NCP",constituency:"AKOLE (ST)",criminal:0,education:"Graduate Professional",assets:""},
  {id:98,name:"Dr. Milind Ramji Narote",party:"BJP",constituency:"GADCHIROLI (ST)",criminal:0,education:"Graduate Professional",assets:""},
  {id:99,name:"Dr. Pankaj Rajesh Bhoyar",party:"BJP",constituency:"WARDHA",criminal:0,education:"Doctorate",assets:""},
  {id:100,name:"Dr. Rahul Vedprakash Patil",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"PARBHANI",criminal:0,education:"Graduate Professional",assets:""},
  {id:101,name:"Dr. Suresh(Bhau) Dagadu Khade",party:"BJP",constituency:"MIRAJ (SC)",criminal:0,education:"Doctorate",assets:""},
  {id:102,name:"Dr. Vinay Vilasrao Kore (Savkar)",party:"Jan Surajya Shakti",constituency:"SHAHUWADI",criminal:0,education:"12th Pass",assets:""},
  {id:103,name:"Dr.Aher Rahul Daulatrao",party:"BJP",constituency:"CHANDWAD",criminal:0,education:"Post Graduate",assets:""},
  {id:104,name:"Dr.Vijaykumar Krushanrao Gavit",party:"BJP",constituency:"NANDURBAR (ST)",criminal:0,education:"Post Graduate",assets:""},
  {id:105,name:"Gaikwad Sanjay Rambhau",party:"Shiv Sena",constituency:"BULDHANA",criminal:0,education:"8th Pass",assets:""},
  {id:106,name:"Gajanan Motiram Lawate",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"DARYAPUR (SC)",criminal:0,education:"8th Pass",assets:""},
  {id:107,name:"Ganesh Ramchandra Naik",party:"BJP",constituency:"AIROLI",criminal:0,education:"10th Pass",assets:""},
  {id:108,name:"Gavit Rajendra Dhedya",party:"Shiv Sena",constituency:"PALGHAR (ST)",criminal:0,education:"Graduate",assets:""},
  {id:109,name:"Girish Dattatray Mahajan",party:"BJP",constituency:"JAMNER",criminal:0,education:"12th Pass",assets:""},
  {id:110,name:"Gogawale Bharat Maruti",party:"Shiv Sena",constituency:"MAHAD",criminal:0,education:"8th Pass",assets:""},
  {id:111,name:"Gopichand Kundlik Padalkar",party:"BJP",constituency:"JAT",criminal:0,education:"12th Pass",assets:""},
  {id:112,name:"Gulabrao Raghunath Patil",party:"Shiv Sena",constituency:"JALGAON RURAL",criminal:0,education:"12th Pass",assets:""},
  {id:113,name:"Harish Marotiappa Pimple",party:"BJP",constituency:"MURTIZAPUR (SC)",criminal:0,education:"12th Pass",assets:""},
  {id:114,name:"Haroon Khan",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"VERSOVA",criminal:0,education:"10th Pass",assets:""},
  {id:115,name:"Hemant Narayan Rasane",party:"BJP",constituency:"KASBA PETH",criminal:0,education:"12th Pass",assets:""},
  {id:116,name:"Hiray Seema Mahesh",party:"BJP",constituency:"NASHIK WEST",criminal:0,education:"Graduate",assets:""},
  {id:117,name:"Indranil Manohar Naik",party:"NCP",constituency:"PUSAD",criminal:0,education:"Graduate Professional",assets:""},
  {id:118,name:"Jadhav Bhaskar Bhaurao",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"GUHAGAR",criminal:0,education:"10th Pass",assets:""},
  {id:119,name:"Jagtap Shankar Pandurang",party:"BJP",constituency:"CHINCHWAD",criminal:0,education:"Post Graduate",assets:""},
  {id:120,name:"Jaiswal Pradeep Shivnarayan",party:"Shiv Sena",constituency:"AURANGABAD CENTRAL",criminal:0,education:"8th Pass",assets:""},
  {id:121,name:"Jaykumar Bhagavanrao Gore",party:"BJP",constituency:"MAN",criminal:0,education:"12th Pass",assets:""},
  {id:122,name:"Jaykumar Jitendrasinh Rawal",party:"BJP",constituency:"SINDHKHEDA",criminal:0,education:"Graduate",assets:""},
  {id:123,name:"Jorgewar Kishor Gajanan",party:"BJP",constituency:"CHANDRAPUR (SC)",criminal:0,education:"10th Pass",assets:""},
  {id:124,name:"Kadam Vishwajit Patangrao",party:"INC",constituency:"PALUS-KADEGAON",criminal:0,education:"Doctorate",assets:""},
  {id:125,name:"Kadam Yogeshdada Ramdas",party:"Shiv Sena",constituency:"DAPOLI",criminal:0,education:"Graduate",assets:""},
  {id:126,name:"Kailas Balasaheb Ghadge Patil",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"OSMANABAD",criminal:0,education:"Graduate Professional",assets:""},
  {id:127,name:"Kalidas Nilkanth Kolambkar",party:"BJP",constituency:"WADALA",criminal:0,education:"10th Pass",assets:""},
  {id:128,name:"Kalyanshetti Sachin Panchappa",party:"BJP",constituency:"AKKALKOT",criminal:0,education:"Graduate Professional",assets:""},
  {id:129,name:"Karan Sanjay Deotale",party:"BJP",constituency:"WARORA",criminal:0,education:"Post Graduate",assets:""},
  {id:130,name:"Kardile Shivaji Bhanudas",party:"BJP",constituency:"RAHURI",criminal:0,education:"8th Pass",assets:""},
  {id:131,name:"Karemore Raju Manikrao",party:"NCP",constituency:"TUMSAR",criminal:0,education:"12th Pass",assets:""},
  {id:132,name:"Kashinath Mahadu Date Sir",party:"NCP",constituency:"PARNER",criminal:0,education:"Graduate",assets:""},
  {id:133,name:"Kashiram Vechan Pawara",party:"BJP",constituency:"SHIRPUR (ST)",criminal:0,education:"Literate",assets:""},
  {id:134,name:"Kayande Manoj Devanand",party:"NCP",constituency:"SINDHKHED RAJA",criminal:0,education:"Graduate",assets:""},
  {id:135,name:"Kewalram Tulsiram Kale",party:"BJP",constituency:"MELGHAT (ST)",criminal:0,education:"10th Pass",assets:""},
  {id:136,name:"Kharat Siddharth Rambhau",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"MEHKAR (SC)",criminal:0,education:"Post Graduate",assets:""},
  {id:137,name:"Khopde Krishna Pancham",party:"BJP",constituency:"NAGPUR EAST",criminal:0,education:"5th Pass",assets:""},
  {id:138,name:"Khoskar Hiraman Bhika",party:"NCP",constituency:"IGATPURI (ST)",criminal:0,education:"Literate",assets:""},
  {id:139,name:"Kiran Alias Bhaiyya Samant",party:"Shiv Sena",constituency:"RAJAPUR",criminal:0,education:"Others",assets:""},
  {id:140,name:"Kisan Maroti Wankhede",party:"BJP",constituency:"UMARKHED (SC)",criminal:0,education:"Graduate Professional",assets:""},
  {id:141,name:"Kisan Shankar Kathore",party:"BJP",constituency:"MURBAD",criminal:0,education:"10th Pass",assets:""},
  {id:142,name:"Kishor Appa Patil",party:"Shiv Sena",constituency:"PACHORA",criminal:0,education:"12th Pass",assets:""},
  {id:143,name:"Kohlikar Baburao Kadam",party:"Shiv Sena",constituency:"HADGAON",criminal:0,education:"Graduate",assets:""},
  {id:144,name:"Kuche Narayan Tilakchand",party:"BJP",constituency:"BADNAPUR (SC)",criminal:0,education:"Graduate",assets:""},
  {id:145,name:"Kul Rahul Subhashrao",party:"BJP",constituency:"DAUND",criminal:0,education:"Graduate Professional",assets:""},
  {id:146,name:"Kute Sanjay Shriram",party:"BJP",constituency:"JALGAON (JAMOD)",criminal:0,education:"Graduate Professional",assets:""},
  {id:147,name:"Madhuri Satish Misal",party:"BJP",constituency:"PARVATI",criminal:0,education:"Graduate",assets:""},
  {id:148,name:"Mahendra Hari Dalvi",party:"Shiv Sena",constituency:"ALIBAG",criminal:0,education:"8th Pass",assets:""},
  {id:149,name:"Mahesh (Dada) Kisan Landge",party:"BJP",constituency:"BHOSARI",criminal:0,education:"Graduate Professional",assets:""},
  {id:150,name:"Mahesh Baldi",party:"BJP",constituency:"URAN",criminal:0,education:"12th Pass",assets:""},
  {id:151,name:"Mahesh Baliram Sawant",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"MAHIM",criminal:0,education:"Graduate",assets:""},
  {id:152,name:"Mahesh Sambhajiraje Shinde",party:"Shiv Sena",constituency:"KOREGAON",criminal:0,education:"Graduate Professional",assets:""},
  {id:153,name:"Manda Vijay Mhatre",party:"BJP",constituency:"BELAPUR",criminal:0,education:"5th Pass",assets:""},
  {id:154,name:"Mangal Prabhat Lodha",party:"BJP",constituency:"MALABAR HILL",criminal:0,education:"Graduate Professional",assets:""},
  {id:155,name:"Mangesh Ramesh Chavan",party:"BJP",constituency:"CHALISGAON",criminal:0,education:"12th Pass",assets:""},
  {id:156,name:"Manjula Tulshiram Gavit",party:"Shiv Sena",constituency:"SAKRI (ST)",criminal:0,education:"Graduate",assets:""},
  {id:157,name:"Manoj Bhimrao Ghorpade",party:"BJP",constituency:"KARAD NORTH",criminal:0,education:"Graduate",assets:""},
  {id:158,name:"Manoj Pandurang Jamsutkar",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"BYCULLA",criminal:0,education:"12th Pass",assets:""},
  {id:159,name:"Mihir Kotecha",party:"BJP",constituency:"MULUND",criminal:0,education:"12th Pass",assets:""},
  {id:160,name:"Mohan Gopalrao Mate",party:"BJP",constituency:"NAGPUR SOUTH",criminal:0,education:"Others",assets:""},
  {id:161,name:"Mrs Shweta Vidyadhar Mahale",party:"BJP",constituency:"CHIKHLI",criminal:0,education:"Others",assets:""},
  {id:162,name:"Mufti Mohammad Ismail Abdul Khalique",party:"All India Majlis-E-Ittehadul Muslimeen",constituency:"MALEGAON CENTRAL",criminal:0,education:"Others",assets:""},
  {id:163,name:"Mungantiwar Sudhir Sacchidanand",party:"BJP",constituency:"BALLARPUR",criminal:0,education:"Post Graduate",assets:""},
  {id:164,name:"Murji Patel (Kaka)",party:"Shiv Sena",constituency:"ANDHERI EAST",criminal:0,education:"8th Pass",assets:""},
  {id:165,name:"Mushrif Hasan Miyalal",party:"NCP",constituency:"KAGAL",criminal:0,education:"Graduate",assets:""},
  {id:166,name:"Mutkule Tanhaji Sakharamji",party:"BJP",constituency:"HINGOLI",criminal:0,education:"10th Pass",assets:""},
  {id:167,name:"Namita Akshay Mundada",party:"BJP",constituency:"KAIJ (SC)",criminal:0,education:"Graduate Professional",assets:""},
  {id:168,name:"Nanabhau Falgunrao Patole",party:"INC",constituency:"SAKOLI",criminal:0,education:"12th Pass",assets:""},
  {id:169,name:"Narayan (Aba) Govindrao Patil",party:"NCP-Sharadchandra Pawar",constituency:"KARMALA",criminal:0,education:"8th Pass",assets:""},
  {id:170,name:"Narendra Mehta",party:"BJP",constituency:"MEERA BHAYANDAR",criminal:0,education:"8th Pass",assets:""},
  {id:171,name:"Nilangekar Sambhaji Diliprao Patil",party:"BJP",constituency:"NILANGA",criminal:0,education:"Others",assets:""},
  {id:172,name:"Nilesh Narayan Rane",party:"Shiv Sena",constituency:"KUDAL",criminal:0,education:"Doctorate",assets:""},
  {id:173,name:"Nitesh Narayan Rane",party:"BJP",constituency:"KANKAVLI",criminal:0,education:"Post Graduate",assets:""},
  {id:174,name:"Nitin Bhikanrao Deshmukh",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"BALAPUR",criminal:0,education:"12th Pass",assets:""},
  {id:175,name:"Nitinbhau Arjun (A.T.) Pawar",party:"NCP",constituency:"KALWAN (ST)",criminal:0,education:"Graduate",assets:""},
  {id:176,name:"Ogale Hemant Bhujangrao",party:"INC",constituency:"SHRIRAMPUR (SC)",criminal:0,education:"Post Graduate",assets:""},
  {id:177,name:"Parag Shah",party:"BJP",constituency:"GHATKOPAR EAST",criminal:0,education:"Graduate",assets:""},
  {id:178,name:"Patil Vikhe Radhakrushna Eknathrao",party:"BJP",constituency:"SHIRDI",criminal:0,education:"Graduate",assets:""},
  {id:179,name:"Prakash (Dada) Sunadarrao Solanke",party:"NCP",constituency:"MAJALGAON",criminal:0,education:"Graduate",assets:""},
  {id:180,name:"Prakash Gunwant Bharsakle",party:"BJP",constituency:"AKOT",criminal:0,education:"8th Pass",assets:""},
  {id:181,name:"Prakash Surve",party:"Shiv Sena",constituency:"MAGATHANE",criminal:0,education:"Graduate",assets:""},
  {id:182,name:"Prashant Ramsheth Thakur",party:"BJP",constituency:"PANVEL",criminal:0,education:"12th Pass",assets:""},
  {id:183,name:"Pratap Baburao Sarnaik",party:"Shiv Sena",constituency:"OVALA MAJIWADA",criminal:0,education:"10th Pass",assets:""},
  {id:184,name:"Prataprao Patil Chikhalikar",party:"NCP",constituency:"LOHA",criminal:0,education:"12th Pass",assets:""},
  {id:185,name:"Pravin Virbhadrayya Swami (Sir)",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"UMARGA (SC)",criminal:0,education:"Post Graduate",assets:""},
  {id:186,name:"Prof. Dr. Ashok Ramaji Wooike",party:"BJP",constituency:"RALEGAON (ST)",criminal:0,education:"Post Graduate",assets:""},
  {id:187,name:"Prof.Dr.Tanaji Jayawant Sawant",party:"Shiv Sena",constituency:"PARANDA",criminal:0,education:"Doctorate",assets:""},
  {id:188,name:"Raghavendra (Ramdada) Manohar Patil",party:"BJP",constituency:"DHULE RURAL",criminal:0,education:"Graduate Professional",assets:""},
  {id:189,name:"Rahul Prakash Awade",party:"BJP",constituency:"ICHALKARANJI",criminal:0,education:"Post Graduate",assets:""},
  {id:190,name:"Rais Kasam Shaikh",party:"SP",constituency:"BHIWANDI EAST",criminal:0,education:"Graduate",assets:""},
  {id:191,name:"Rajan Balkrishna Naik",party:"BJP",constituency:"NALASOPARA",criminal:0,education:"Graduate",assets:""},
  {id:192,name:"Rajendra Shamgonda Patil (Yadravkar)",party:"Rajarshi Shahu Vikas Aghadi",constituency:"SHIROL",criminal:0,education:"Others",assets:""},
  {id:193,name:"Rajesh Govardhan More",party:"Shiv Sena",constituency:"KALYAN RURAL",criminal:0,education:"Graduate",assets:""},
  {id:194,name:"Rajesh Sambhajirao Pawar",party:"BJP",constituency:"NAIGAON",criminal:0,education:"Graduate Professional",assets:""},
  {id:195,name:"Rajesh Shriramji Wankhade",party:"BJP",constituency:"TEOSA",criminal:0,education:"Graduate",assets:""},
  {id:196,name:"Rajesh Udesing Padvi",party:"BJP",constituency:"SHAHADA (ST)",criminal:0,education:"Graduate",assets:""},
  {id:197,name:"Rajesh Uttamrao Vitekar",party:"NCP",constituency:"PATHRI",criminal:0,education:"Graduate",assets:""},
  {id:198,name:"Rajesh Vinayak Kshirsagar",party:"Shiv Sena",constituency:"KOLHAPUR NORTH",criminal:0,education:"Graduate",assets:""},
  {id:199,name:"Raju Narayan Todsam",party:"BJP",constituency:"ARNI (ST)",criminal:0,education:"Post Graduate",assets:""},
  {id:200,name:"Ram Kadam",party:"BJP",constituency:"GHATKOPAR WEST",criminal:0,education:"10th Pass",assets:""},
  {id:201,name:"Ramesh Kashiram Karad",party:"BJP",constituency:"LATUR RURAL",criminal:0,education:"Graduate Professional",assets:""},
  {id:202,name:"Ranajagjitsinha Padmasinha Patil",party:"BJP",constituency:"TULJAPUR",criminal:0,education:"Graduate Professional",assets:""},
  {id:203,name:"Randhir Pralhadrao Sawarkar",party:"BJP",constituency:"AKOLA EAST",criminal:0,education:"Graduate Professional",assets:""},
  {id:204,name:"Ranjanatai (Sanjana) Harshvardhan Jadhav",party:"Shiv Sena",constituency:"KANNAD",criminal:0,education:"Graduate",assets:""},
  {id:205,name:"Rathod Sanjay Dulichand",party:"Shiv Sena",constituency:"DIGRAS",criminal:0,education:"Graduate Professional",assets:""},
  {id:206,name:"Ravi Gangadhar Rana",party:"Rashtriya Yuva Swabhiman Party",constituency:"BADNERA",criminal:0,education:"12th Pass",assets:""},
  {id:207,name:"Ravisheth Patil",party:"BJP",constituency:"PEN",criminal:0,education:"12th Pass",assets:""},
  {id:208,name:"Rohit Pawar",party:"NCP-Sharadchandra Pawar",constituency:"KARJAT JAMKHED",criminal:0,education:"Graduate Professional",assets:""},
  {id:209,name:"Sachin Patil",party:"NCP",constituency:"PHALTAN (SC)",criminal:0,education:"8th Pass",assets:""},
  {id:210,name:"Sai Prakash Dahake",party:"BJP",constituency:"KARANJA",criminal:0,education:"12th Pass",assets:""},
  {id:211,name:"Sajid Khan Pathan",party:"INC",constituency:"AKOLA WEST",criminal:0,education:"8th Pass",assets:""},
  {id:212,name:"Sameer Dattatraya Meghe",party:"BJP",constituency:"HINGNA",criminal:0,education:"Graduate",assets:""},
  {id:213,name:"Samir Tryambakrao Kunawar",party:"BJP",constituency:"HINGANGHAT",criminal:0,education:"12th Pass",assets:""},
  {id:214,name:"Sana Malik",party:"NCP",constituency:"ANUSHAKTI NAGAR",criminal:0,education:"Graduate Professional",assets:""},
  {id:215,name:"Sandeep Ravindra Kshirsagar",party:"NCP-Sharadchandra Pawar",constituency:"BEED",criminal:0,education:"Graduate",assets:""},
  {id:216,name:"Sangram Arunkaka Jagtap",party:"NCP",constituency:"AHMEDNAGAR CITY",criminal:0,education:"Graduate",assets:""},
  {id:217,name:"Sanjay Govind Potnis",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"KALINA",criminal:0,education:"10th Pass",assets:""},
  {id:218,name:"Sanjay Mukund Kelkar",party:"BJP",constituency:"THANE",criminal:0,education:"Graduate",assets:""},
  {id:219,name:"Sanjay Narayanrao Meshram",party:"INC",constituency:"UMRED (SC)",criminal:0,education:"Graduate Professional",assets:""},
  {id:220,name:"Sanjay Pandurang Shirsat",party:"Shiv Sena",constituency:"AURANGABAD WEST (SC)",criminal:0,education:"12th Pass",assets:""},
  {id:221,name:"Sanjay Puram",party:"BJP",constituency:"AMGAON (ST)",criminal:0,education:"12th Pass",assets:""},
  {id:222,name:"Sanjay Upadhyay",party:"BJP",constituency:"BORIVALI",criminal:0,education:"Post Graduate",assets:""},
  {id:223,name:"Savkare Sanjay Waman",party:"BJP",constituency:"BHUSAWAL (SC)",criminal:0,education:"Others",assets:""},
  {id:224,name:"Shankar Hiraman Mandekar",party:"NCP",constituency:"BHOR",criminal:0,education:"12th Pass",assets:""},
  {id:225,name:"Sharaddada Bhimaji Sonawane",party:"IND",constituency:"JUNNAR",criminal:0,education:"10th Pass",assets:""},
  {id:226,name:"Shekhar Govindrao Nikam",party:"NCP",constituency:"CHIPLUN",criminal:0,education:"Post Graduate",assets:""},
  {id:227,name:"Shirishkumar Surupsing Naik",party:"INC",constituency:"NAWAPUR (ST)",criminal:0,education:"Graduate",assets:""},
  {id:228,name:"Shivaji Shattupa Patil",party:"IND",constituency:"CHANDGAD",criminal:0,education:"12th Pass",assets:""},
  {id:229,name:"Shivendraraje Abhaysinhraje Bhonsle",party:"BJP",constituency:"SATARA",criminal:0,education:"12th Pass",assets:""},
  {id:230,name:"Shyam Ramcharan Khode",party:"BJP",constituency:"WASHIM (SC)",criminal:0,education:"10th Pass",assets:""},
  {id:231,name:"Siddharth Anil Shirole",party:"BJP",constituency:"SHIVAJINAGAR",criminal:0,education:"Graduate Professional",assets:""},
  {id:232,name:"Sneha Dube Pandit",party:"BJP",constituency:"VASAI",criminal:0,education:"Graduate Professional",assets:""},
  {id:233,name:"Sulbha Ganpat Gaikwad",party:"BJP",constituency:"KALYAN EAST",criminal:0,education:"5th Pass",assets:""},
  {id:234,name:"Sulbha Sanjay Khodke",party:"NCP",constituency:"AMRAVATI",criminal:0,education:"Post Graduate",assets:""},
  {id:235,name:"Sumit Wankhede",party:"BJP",constituency:"ARVI",criminal:0,education:"Graduate Professional",assets:""},
  {id:236,name:"Sunil Rajaram Raut",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"VIKHROLI",criminal:0,education:"12th Pass",assets:""},
  {id:237,name:"Sunil Shankarrao Shelke",party:"NCP",constituency:"MAVAL",criminal:0,education:"8th Pass",assets:""},
  {id:238,name:"Sunil Waman Prabhu",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"DINDOSHI",criminal:0,education:"12th Pass",assets:""},
  {id:239,name:"Suresh Damu Bhole (Raju Mama)",party:"BJP",constituency:"JALGAON CITY",criminal:0,education:"Graduate",assets:""},
  {id:240,name:"Thorve Mahendra Sadashiv",party:"Shiv Sena",constituency:"KARJAT",criminal:0,education:"8th Pass",assets:""},
  {id:241,name:"Tushar Govindrao Rathod",party:"BJP",constituency:"MUKHED",criminal:0,education:"Graduate Professional",assets:""},
  {id:242,name:"Uday Ravindra Samant",party:"Shiv Sena",constituency:"RATNAGIRI",criminal:0,education:"Others",assets:""},
  {id:243,name:"Udhan Hikmat Baliram",party:"Shiv Sena",constituency:"GANSAVANGI",criminal:0,education:"Graduate Professional",assets:""},
  {id:244,name:"Umesh Alias Chandu Atmaramji Yawalkar",party:"BJP",constituency:"MORSHI",criminal:0,education:"Graduate Professional",assets:""},
  {id:245,name:"Varun Satish Sardesai",party:"Shiv Sena (Uddhav Balasaheb Thackeray)",constituency:"VANDRE EAST",criminal:0,education:"Post Graduate",assets:""},
  {id:246,name:"Vidya Jaiprakash Thakur",party:"BJP",constituency:"GOREGAON",criminal:0,education:"8th Pass",assets:""},
  {id:247,name:"Vijay Bharatlal Rahangdale",party:"BJP",constituency:"TIRORA",criminal:0,education:"Graduate",assets:""},
  {id:248,name:"Vijay Namdeorao Wadettiwar",party:"INC",constituency:"BRAMHAPURI",criminal:0,education:"10th Pass",assets:""},
  {id:249,name:"Vijaysinh Shivajirao Pandit",party:"NCP",constituency:"GEORAI",criminal:0,education:"Graduate Professional",assets:""},
  {id:250,name:"Vikas Pandurang Thakre",party:"INC",constituency:"NAGPUR WEST",criminal:0,education:"Graduate",assets:""},
  {id:251,name:"Vilas Sukur Tare",party:"Shiv Sena",constituency:"BOISAR (ST)",criminal:0,education:"5th Pass",assets:""},
  {id:252,name:"Vinod Bhiva Nikole",party:"CPI(M)",constituency:"DAHANU (ST)",criminal:0,education:"12th Pass",assets:""},
  {id:253,name:"Vishwanath Atmaram Bhoir",party:"Shiv Sena",constituency:"KALYAN WEST",criminal:0,education:"Graduate",assets:""},
  {id:254,name:"Vitthal Vakilrao Langhe",party:"Shiv Sena",constituency:"NEVASA",criminal:0,education:"8th Pass",assets:""},
  {id:255,name:"Yogesh Sagar",party:"BJP",constituency:"CHARKOP",criminal:0,education:"10th Pass",assets:""},
]

const PARTY_COLORS = {
  'BJP':    { bg:'#fde8e7', color:'#c0392b', dot:'#c0392b' },
  'INC':    { bg:'#e0f5ec', color:'#0a6640', dot:'#0a8754' },
  'Shiv Sena': { bg:'#fff8e1', color:'#92400e', dot:'#d97706' },
  'Shiv Sena (Uddhav Balasaheb Thackeray)': { bg:'#fef3c7', color:'#78350f', dot:'#f59e0b' },
  'NCP':    { bg:'#ddf0fc', color:'#0369a1', dot:'#0ea5e9' },
  'NCP-Sharadchandra Pawar': { bg:'#dde8f5', color:'#1e40af', dot:'#3b82f6' },
  'IND':    { bg:'#ecf0f1', color:'#4a5568', dot:'#718096' },
  'SP':     { bg:'#fce7f3', color:'#9d174d', dot:'#ec4899' },
  'All India Majlis-E-Ittehadul Muslimeen': { bg:'#e0f5ec', color:'#065f46', dot:'#059669' },
  'Jan Surajya Shakti': { bg:'#f5f3ff', color:'#5b21b6', dot:'#7c3aed' },
  'BSP':    { bg:'#f3e8fd', color:'#6d28d9', dot:'#8b5cf6' },
  'CPI(M)': { bg:'#fde8e7', color:'#7f1d1d', dot:'#dc2626' },
  'Peasants And Workers Party of India': { bg:'#ecfdf5', color:'#064e3b', dot:'#10b981' },
};
const AVATAR_PALETTES = [
  ['#bfd7ea','#0a3d5c'],['#ddf0fc','#073d52'],['#fde8e7','#7c1d1d'],
  ['#fff3e0','#7c2d12'],['#f3e8fd','#4c1d95'],['#e0f5ec','#064e3b'],
  ['#fef3c7','#451a03'],['#dde8f5','#1e3a5f'],
];
function getAvatarStyle(n){return AVATAR_PALETTES[n.charCodeAt(0)%AVATAR_PALETTES.length];}
function initials(n){const p=n.trim().split(' ');return p.length===1?p[0].slice(0,2).toUpperCase():(p[0][0]+p[p.length-1][0]).toUpperCase();}
function getPartyStyle(p){return PARTY_COLORS[p]||{bg:'#f5f5f5',color:'#555',dot:'#888'};}
function crimClass(n){const c=parseInt(n||0);if(c===0)return['criminal-none','✓ Clean'];if(c<=4)return['criminal-low',`⚠ ${c} Case${c>1?'s':''}`];return['criminal-high',`⛔ ${c} Cases`];}

function buildDashboard(){
  const total=CANDIDATES.length;
  const parties=new Set(CANDIDATES.map(c=>c.party)).size;
  const clean=CANDIDATES.filter(c=>parseInt(c.criminal||0)===0).length;
  const grads=CANDIDATES.filter(c=>['Graduate','Graduate Professional','Post Graduate','Doctorate'].includes(c.education)).length;
  document.getElementById('hs-parties').textContent=parties;
  document.getElementById('hs-clean').textContent=clean;
  document.getElementById('hs-grads').textContent=grads;
  const constituencies=new Set(CANDIDATES.map(c=>c.constituency)).size;
  const withCriminal=CANDIDATES.filter(c=>parseInt(c.criminal||0)>0).length;
  const docs=CANDIDATES.filter(c=>c.education==='Doctorate').length;
  const highCrim=CANDIDATES.filter(c=>parseInt(c.criminal||0)>=5).length;
  const sg=document.getElementById('stat-grid');
  const stats=[
    {icon:'👥',color:'rgba(191,215,234,.3)',num:total,lbl:'Total Candidates'},
    {icon:'🏛️',color:'rgba(80,140,164,.2)',num:parties,lbl:'Political Parties'},
    {icon:'✅',color:'rgba(10,135,84,.15)',num:clean,lbl:'Clean Records'},
    {icon:'🗺️',color:'rgba(145,174,193,.3)',num:constituencies,lbl:'Constituencies'},
    {icon:'🎓',color:'rgba(80,140,164,.15)',num:grads,lbl:'Graduate & Above'},
    {icon:'⚖️',color:'rgba(192,57,43,.1)',num:withCriminal,lbl:'Have Criminal Cases'},
    {icon:'🔬',color:'rgba(108,91,163,.1)',num:docs,lbl:'Doctorate Holders'},
    {icon:'🚨',color:'rgba(192,57,43,.15)',num:highCrim,lbl:'5+ Criminal Cases'},
  ];
  sg.innerHTML=stats.map(s=>`<div class="stat-card"><div class="stat-icon" style="background:${s.color}">${s.icon}</div><div class="stat-num">${s.num}</div><div class="stat-lbl">${s.lbl}</div></div>`).join('');
  const partyCounts={};
  CANDIDATES.forEach(c=>partyCounts[c.party]=(partyCounts[c.party]||0)+1);
  const topParties=Object.entries(partyCounts).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const maxP=topParties[0][1];
  const pColors=['#c0392b','#0a8754','#d97706','#f59e0b','#0369a1','#1e40af','#6d28d9','#4a5568'];
  document.getElementById('party-chart').innerHTML=topParties.map(([p,n],i)=>`<div class="bar-row"><div class="bar-label" title="${p}">${p.length>13?p.slice(0,13)+'…':p}</div><div class="bar-track"><div class="bar-fill" style="width:${(n/maxP*100).toFixed(1)}%;background:${pColors[i]}"></div></div><div class="bar-val">${n}</div></div>`).join('');
  const eduOrder=['Doctorate','Post Graduate','Graduate Professional','Graduate','12th Pass','10th Pass','8th Pass','Literate'];
  const eduCounts={};CANDIDATES.forEach(c=>eduCounts[c.education]=(eduCounts[c.education]||0)+1);
  const maxE=Math.max(...eduOrder.map(e=>eduCounts[e]||0));
  const eColors=['#004f2d','#0a8754','#508ca4','#91aec1','#bfd7ea','#508ca4','#0a8754','#004f2d'];
  document.getElementById('edu-chart').innerHTML=eduOrder.map((e,i)=>`<div class="bar-row"><div class="bar-label">${e}</div><div class="bar-track"><div class="bar-fill" style="width:${((eduCounts[e]||0)/maxE*100).toFixed(1)}%;background:${eColors[i]}"></div></div><div class="bar-val">${eduCounts[e]||0}</div></div>`).join('');
}

let filtered=[...CANDIDATES],currentPage=1;
const PER_PAGE=24;
function buildFilters(){
  const parties=[...new Set(CANDIDATES.map(c=>c.party))].sort();
  const pf=document.getElementById('partyFilter');
  parties.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p.length>30?p.slice(0,30)+'…':p;pf.appendChild(o);});
  const edus=[...new Set(CANDIDATES.map(c=>c.education))].filter(Boolean).sort();
  const ef=document.getElementById('eduFilter');
  edus.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;ef.appendChild(o);});
}
function filterCandidates(){
  const q=document.getElementById('searchInput').value.toLowerCase();
  const party=document.getElementById('partyFilter').value;
  const edu=document.getElementById('eduFilter').value;
  const crim=document.getElementById('crimFilter').value;
  filtered=CANDIDATES.filter(c=>{
    if(q&&!c.name.toLowerCase().includes(q)&&!c.party.toLowerCase().includes(q)&&!(c.constituency||'').toLowerCase().includes(q))return false;
    if(party&&c.party!==party)return false;
    if(edu&&c.education!==edu)return false;
    const cn=parseInt(c.criminal||0);
    if(crim==='0'&&cn!==0)return false;
    if(crim==='1'&&(cn<1||cn>=5))return false;
    if(crim==='5'&&cn<5)return false;
    return true;
  });
  currentPage=1;renderCards();
}
function renderCards(){
  const grid=document.getElementById('candidates-grid');
  const start=(currentPage-1)*PER_PAGE;
  const slice=filtered.slice(start,start+PER_PAGE);
  document.getElementById('results-count').textContent=`${filtered.length} candidates`;
  if(slice.length===0){grid.innerHTML=`<div style="grid-column:1/-1;text-align:center;padding:3rem;color:var(--ink-faint)">No candidates found.</div>`;document.getElementById('pagination').innerHTML='';return;}
  grid.innerHTML=slice.map(c=>{
    const [ab,ac]=getAvatarStyle(c.name);
    const ps=getPartyStyle(c.party);
    const [cc,cl]=crimClass(c.criminal);
    const sp=c.party.length>22?c.party.slice(0,22)+'…':c.party;
    return`<div class="candidate-card" onclick="openModal(${c.id})">
      <div class="card-top"><div class="avatar" style="background:${ab};color:${ac}">${initials(c.name)}</div><div><div class="card-name">${c.name}</div><div class="constituency-tag">📍 ${c.constituency||'—'}</div></div></div>
      <div class="party-badge" style="background:${ps.bg};color:${ps.color}"><span style="width:5px;height:5px;border-radius:50%;background:${ps.dot};flex-shrink:0;display:inline-block"></span>${sp}</div>
      <div class="card-grid"><div><div class="card-field-label">Education</div><div class="card-field-value">${c.education||'—'}</div></div><div><div class="card-field-label">Criminal</div><div class="card-field-value"><span class="criminal-badge ${cc}">${cl}</span></div></div></div>
    </div>`;
  }).join('');
  renderPagination();
}
function renderPagination(){
  const total=Math.ceil(filtered.length/PER_PAGE);
  const pag=document.getElementById('pagination');
  if(total<=1){pag.innerHTML='';return;}
  let btns=`<button class="page-btn" onclick="goPage(${currentPage-1})" ${currentPage===1?'disabled':''}>‹</button>`;
  for(let i=1;i<=total;i++){
    if(i===1||i===total||Math.abs(i-currentPage)<=2)btns+=`<button class="page-btn ${i===currentPage?'active':''}" onclick="goPage(${i})">${i}</button>`;
    else if(Math.abs(i-currentPage)===3)btns+=`<span style="padding:.5rem .3rem;color:var(--ink-faint)">…</span>`;
  }
  btns+=`<button class="page-btn" onclick="goPage(${currentPage+1})" ${currentPage===total?'disabled':''}>›</button>`;
  pag.innerHTML=btns;
}
function goPage(p){const total=Math.ceil(filtered.length/PER_PAGE);if(p<1||p>total)return;currentPage=p;renderCards();document.getElementById('candidates-grid').scrollIntoView({behavior:'smooth',block:'start'});}

function openModal(id){
  const c=CANDIDATES.find(x=>x.id==id);if(!c)return;
  const [ab,ac]=getAvatarStyle(c.name);
  const ps=getPartyStyle(c.party);
  const [cc,cl]=crimClass(c.criminal);
  document.getElementById('modal-content').innerHTML=`
    <div class="modal-avatar" style="background:${ab};color:${ac}">${initials(c.name)}</div>
    <div class="modal-name">${c.name}</div>
    <div class="modal-party"><span class="party-badge" style="background:${ps.bg};color:${ps.color}"><span style="width:5px;height:5px;border-radius:50%;background:${ps.dot};display:inline-block"></span>${c.party}</span></div>
    <div class="modal-grid">
      <div class="modal-field"><div class="modal-field-label">Constituency</div><div class="modal-field-value">${c.constituency||'—'}</div></div>
      <div class="modal-field"><div class="modal-field-label">Education</div><div class="modal-field-value">${c.education||'—'}</div></div>
      <div class="modal-field"><div class="modal-field-label">Criminal Cases</div><div class="modal-field-value"><span class="criminal-badge ${cc}">${cl}</span></div></div>
      <div class="modal-field"><div class="modal-field-label">Candidate #</div><div class="modal-field-value">#${c.id}</div></div>
      <div class="modal-field" style="grid-column:1/-1"><div class="modal-field-label">Total Assets</div><div class="modal-field-value" style="color:var(--c3)">${c.assets||'Not Disclosed'}</div></div>
    </div>`;
  document.getElementById('modal-overlay').classList.add('open');
  document.body.style.overflow='hidden';
}
function closeModal(e){if(e&&e.target!==document.getElementById('modal-overlay'))return;closeModalDirect();}
function closeModalDirect(){document.getElementById('modal-overlay').classList.remove('open');document.body.style.overflow='';}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModalDirect();});

function showPage(name,btn){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if(btn)btn.classList.add('active');
  if(name==='candidates'&&document.getElementById('candidates-grid').innerHTML===''){buildFilters();filterCandidates();}
}

async function checkApiStatus(){
  const el=document.getElementById('model-status-indicator');
  if(!el)return;
  el.textContent='⏳ Checking…';el.style.color='var(--ink-faint)';
  try{
    const r=await fetch('http://127.0.0.1:5000/models/status',{method:'GET',signal:AbortSignal.timeout(3000)});
    if(r.ok){
      const d=await r.json();
      const hasModel=d.available_models&&d.available_models.includes('maharashtra_voter_prediction.pkl');
      if(hasModel){
        el.style.color='var(--c4)';
        el.innerHTML='✅ <strong>Server connected</strong> — <code style="font-size:.75rem;background:rgba(10,135,84,.1);padding:.1rem .3rem;border-radius:3px;">maharashtra_voter_prediction.pkl</code> loaded and ready.';
      } else {
        el.style.color='#b45309';
        el.innerHTML='⚠️ Server running but model not found.<br><span style="font-size:.75rem">Place <code style="background:rgba(180,83,9,.1);padding:.1rem .3rem;border-radius:3px;">maharashtra_voter_prediction.pkl</code> in the <code style="background:rgba(180,83,9,.1);padding:.1rem .3rem;border-radius:3px;">models/</code> folder and restart the server.</span>';
      }
    } else {
      el.style.color='#c0392b';el.textContent='✗ Server error: '+r.status;
    }
  }catch(err){
    el.style.color='#c0392b';
    el.innerHTML='✗ <strong>Not connected</strong> — run: <code style="font-size:.72rem;background:rgba(192,57,43,.08);padding:.1rem .3rem;border-radius:3px;">uvicorn api:app --reload --port 5000</code>';
  }
}

async function runPrediction(){
  const btn=document.getElementById('pred-run-btn');
  const inputData={
    state:'maharashtra',
    Age:parseInt(document.getElementById('pred-age').value, 10),
    District:document.getElementById('pred-district').value,
    Gender:document.getElementById('pred-gender').value,
    Geography:document.getElementById('pred-geography').value,
    Caste:document.getElementById('pred-caste').value,
    Occupation:document.getElementById('pred-occupation').value
  };
  if(!inputData.Age||isNaN(inputData.Age)||Object.entries(inputData).filter(([k])=>k!=='state'&&k!=='Age').some(([,v])=>!v)){showResultError('Please fill in all demographic fields.');return;}
  btn.textContent='Analyzing…';btn.disabled=true;
  try{
    const response=await fetch('http://127.0.0.1:5000/maharashtra/predict_voter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(inputData)});
    const data=await response.json();
    if(data.status==='success'){
      const rb=document.getElementById('prediction-result');
      const ps=getPartyStyle(data.predicted_party);
      rb.style.background=ps.bg;rb.style.borderColor=ps.dot;rb.style.display='block';
      document.getElementById('predicted-party-text').textContent=data.predicted_party;
      document.getElementById('predicted-party-text').style.color=ps.color;
      document.getElementById('result-sub-text').textContent='Based on the demographic profile entered';
      document.getElementById('result-sub-text').style.color=ps.color;
      if(data.probabilities){
        const sorted=Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).slice(0,5);
        document.getElementById('proba-bars').innerHTML=sorted.map(([party,pct])=>`<div class="proba-row"><div class="proba-label">${party.length>12?party.slice(0,12)+'…':party}</div><div class="proba-track"><div class="proba-fill" style="width:${pct}%;background:${getPartyStyle(party).dot}"></div></div><div class="proba-pct">${pct}%</div></div>`).join('');
      }
    }else{showResultError('Prediction error: '+data.message);}
  }catch(err){showResultError('Could not connect to the ML server.\nMake sure the FastAPI server is running on port 5000.');}
  finally{btn.textContent='🔮 Predict Voting Preference';btn.disabled=false;}
}
function showResultError(msg){
  const rb=document.getElementById('prediction-result');
  rb.style.background='#fde8e7';rb.style.borderColor='#c0392b';rb.style.display='block';
  document.getElementById('predicted-party-text').textContent='Error';
  document.getElementById('predicted-party-text').style.color='#c0392b';
  document.getElementById('result-sub-text').textContent=msg;
  document.getElementById('result-sub-text').style.color='#c0392b';
  document.getElementById('proba-bars').innerHTML='';
}

buildDashboard();
checkApiStatus();