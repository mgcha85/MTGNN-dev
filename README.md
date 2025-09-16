# MTGNN
This is a PyTorch implementation of the paper: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), published in KDD-2020.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
## Data Preparation
### Multivariate time series datasets

Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

### Traffic datasets
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 

```

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Model Training

### Single-step

* Solar-Energy

```
python train_single_step.py --save ./model-solar-3.pt --data ./data/solar_AL.txt --num_nodes 137 --batch_size 4 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-solar-sampling-3.pt --data ./data/solar_AL.txt --num_nodes 137 --batch_size 16 --epochs 30 --horizon 3
```
* Traffic 

```
python train_single_step.py --save ./model-traffic3.pt --data ./data/traffic.txt --num_nodes 862 --batch_size 16 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-traffic-sampling-3.pt --data ./data/traffic --num_nodes 321 --batch_size 16 --epochs 30 --horizon 3
```

* Electricity

```
python train_single_step.py --save ./model-electricity-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 4 --epochs 30 --horizon 3
#sampling 
python train_single_step.py --num_split 3 --save ./model-electricity-sampling-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 16 --epochs 30 --horizon 3
```

* Exchange-Rate

```
python train_single_step.py --save ./model/model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --subgraph_size 8  --batch_size 4 --epochs 30 --horizon 3
#sampling
python train_single_step.py --num_split 3 --save ./model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --subgraph_size 2  --batch_size 16 --epochs 30 --horizon 3
```
### Multi-step
* METR-LA

```
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207
```
* PEMS-BAY

```
python train_multi_step.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY/ --num_nodes 325
```

## Citation

```
@inproceedings{wu2020connecting,
  title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```



## 입력 데이터
data.txt - 각종 보안 위협의 시간별 횟수
data.csv - data.txt에 라벨링한 데이터
graph.csv - data.csv의 컬럼간의 관계성 메타 데이터


## 국가 (37개)

| 코드 | 국가 / 지역 이름 |
|------|-------------------|
| US   | United States (미국) |
| GB   | United Kingdom (영국) |
| CA   | Canada (캐나다) |
| AU   | Australia (호주) |
| UA   | Ukraine (우크라이나) |
| RU   | Russia (러시아) |
| FR   | France (프랑스) |
| DE   | Germany (독일) |
| BR   | Brazil (브라질) |
| CN   | China (중국) |
| JP   | Japan (일본) |
| PK   | Pakistan (파키스탄) |
| KP   | North Korea (북한) |
| KR   | South Korea (대한민국) |
| IN   | India (인도) |
| TW   | Taiwan (대만) |
| NL   | Netherlands (네덜란드) |
| ES   | Spain (스페인) |
| SE   | Sweden (스웨덴) |
| MX   | Mexico (멕시코) |
| IR   | Iran (이란) |
| IL   | Israel (이스라엘) |
| SA   | Saudi Arabia (사우디아라비아) |
| SY   | Syria (시리아) |
| FI   | Finland (핀란드) |
| IE   | Ireland (아일랜드) |
| AT   | Austria (오스트리아) |
| NO   | Norway (노르웨이) |
| CH   | Switzerland (스위스) |
| IT   | Italy (이탈리아) |
| MY   | Malaysia (말레이시아) |
| EG   | Egypt (이집트) |
| TR   | Turkey (터키) |
| PT   | Portugal (포르투갈) |
| PS   | Palestine (팔레스타인) |
| AE   | United Arab Emirates (아랍에미리트) |
| ?    | Unknown / Undefined (알 수 없음) |

## 위협 (26개)
- DDoS
- Phishing
- Ransomware
- Password Attack
- SQL Injection
- Account Hijacking
- Defacement
- Trojan
- Vulnerability
- Zero‑day
- Advanced persistent threat
- XSS
- Malware
- Data Breach
- Disinformation/Misinformation
- Targeted Attack
- Adware
- Brute Force Attack
- Malvertising
- Backdoor
- Botnet
- Cryptojacking
- Worms
- Spyware
- Unknown
- Others


## 대응기술 (98개)
| Solution |
|----------|
| DATA PROVENANCE |
| FORMAL VERIFICATION |
| SIEM |
| STANDARDIZED COMMUNICATION |
| HIDDEN MARKOV MODEL |
| FILE INTEGRITY MONITORING |
| DATA LOSS PREVENTION |
| Identity‑Based Encryption (IBE) |
| IDENTITY MANAGEMENT |
| BIOMETRICS |
| IMAGE RECOGNITION |
| DYNAMIC BINARY INSTRUMENTATION |
| SOURCE IDENTIFICATION |
| HONEYPOT |
| IDS/IPS |
| GRAPHICAL MODEL |
| SECURE SIMPLE PAIRING |
| PASSWORD POLICY |
| LIVENESS DETECTION |
| CRYPTOGRAPHY |
| TAINT ANALYSIS |
| BEHAVIOR BASED DETECTION |
| MULTI FACTOR AUTHENTICATION |
| DATA LEAKAGE DETECTION/PREVENTION |
| KEYSTROKE DYNAMICS |
| MOVING TARGET DEFENSE |
| ADVERSARIAL TRAINING |
| DISTRIBUTED LEDGERS |
| APPLICATION WHITELISTING |
| TRUSTWORTHY AI |
| GAME THEORY |
| DIGITAL WATERMARK |
| DYNAMIC RESOURCE MANAGEMENT |
| USER BEHAVIOR ANALYTICS |
| CAPTCHA |
| HYPERGAME |
| SANDBOXING |
| DATA BACKUPS |
| SSL/TLS |
| DEFENSIVE DISTILLATION |
| PASSWORD HASHING |
| SPATIAL SMOOTHING |
| DIMENSIONALITY REDUCTION |
| CERTIFICATE PINNING |
| RATE LIMITING |
| NETWORK SEGMENTATION |
| DYNAMIC ANALYSIS |
| VULNERABILITY MANAGEMENT |
| NOISE INJECTION |
| MERKLE SIGNATURE |
| BLACKLISTING |
| OUTLIER DETECTION |
| PUBLIC KEY INFRASTRUCTURE |
| DECEPTION TECHNOLOGY |
| ACTIVITY MONITORING |
| VULNERABILITY SCANNER |
| HTTPS |
| SPLIT MANUFACTURING |
| GRAPHICAL AUTHENTICATION |
| ML/DL |
| PRIVACY PRESERVING |
| PENETRATION TESTING |
| VIRTUAL KEYBOARDS |
| PASSWORD MANAGEMENT |
| ANOMALY DETECTION |
| BLACKHOLING |
| PACKET FILTERING |
| DNSSEC |
| VULNERABILITY ASSESSMENT |
| SOFTWARE DEFINED NETWORK |
| SUPPLY CHAIN RISK MANAGEMENT |
| TRAFFIC SHAPING |
| RANK CORRELATION |
| STATIC ANALYSIS |
| CONTINUOUS AUTHENTICATION |
| Bayesian Network |
| PATCH MANAGEMENT |
| VPN |
| SECURE BOOT |
| RRAM |
| SESSION MANAGEMENT |
| Control Flow Integrity |
| STRONG AUTHENTICATION |
| CODE SIGNING |
| RISK ASSESSMENT |
| ACCESS CONTROL |
| BLOCKCHAIN |
| ATTACK TREE |
| 3D FACE RECONSTRUCTION |
| DATA AUGMENTATION |
| MUTUAL AUTHENTICATION |
| PASSWORD STRENGTH METERS |
| DARKNET MONITORING |
| ONE TIME PASSWORD |
| DATA SANITIZATION |
| LEAST PRIVILEGE |
| NLP/LLM |
| ENCRYPTION |





items:
  - "DDoS-ALL"
  - "Phishing-ALL"
  - "Ransomware-ALL"
  - "Password Attack-ALL"
  - "Account Hijacking-ALL"
  - "Trojan-ALL"
  - "Vulnerability-ALL"
  - "Zero-day-ALL"
  - "Advanced persistent threat-ALL"
  - "Malware-ALL"
  - "Disinformation/Misinformation-ALL"
  - "Targeted Attack-ALL"
  - "Brute Force Attack-ALL"
  - "Backdoor-ALL"
  - "Botnet-ALL"
  - "Cryptojacking-ALL"
  - "Mentions-DDoS"
  - "Mentions-Phishing"
  - "Mentions-Ransomware"
  - "Mentions-Password Attack"
  - "Mentions-Account Hijacking"
  - "Mentions-Trojan"
  - "Mentions-Vulnerability"
  - "Mentions-Zero-day"
  - "Mentions-Advanced persistent threat"
  - "Mentions-Malware"
  - "Mentions-Disinformation/Misinformation"
  - "Mentions-Targeted Attack"
  - "Mentions-Brute Force Attack"
  - "Mentions-Backdoor"
  - "Mentions-Botnet"
  - "Mentions-Cryptojacking"
  - "Mentions-MITM"
  - "Mentions-DNS Spoofing"
  - "Mentions-Dropper"
  - "Mentions-Insider Threat"
  - "Mentions-Adversarial Attack"
  - "Mentions-Data Poisoning"
  - "Mentions-Deepfake"
  - "Mentions-Supply Chain"
  - "Mentions-IoT Device Attack"
  - "Mentions-Session Hijacking"
  - "WAR/CONFLICT ALL"
  - "Holidays"
  - "Solution_BLOCKCHAIN_Mentions"
  - "Solution_ACCESS CONTROL_Mentions"
  - "Solution_ENCRYPTION_Mentions"
  - "Solution_SUPPLY CHAIN RISK MANAGEMENT_Mentions"
  - "Solution_IDENTITY MANAGEMENT_Mentions"
  - "Solution_ML/DL_Mentions"
  - "Solution_ANOMALY DETECTION_Mentions"
  - "Solution_CRYPTOGRAPHY_Mentions"
  - "Solution_PENETRATION TESTING_Mentions"
  - "Solution_IDS/IPS_Mentions"
  - "Solution_STATIC ANALYSIS_Mentions"
  - "Solution_DYNAMIC ANALYSIS_Mentions"
  - "Solution_MULTI FACTOR AUTHENTICATION_Mentions"
  - "Solution_LEAST PRIVILEGE_Mentions"
  - "Solution_SESSION MANAGEMENT_Mentions"
  - "Solution_CAPTCHA_Mentions"
  - "Solution_BLACKLISTING_Mentions"
  - "Solution_RATE LIMITING_Mentions"
  - "Solution_GRAPHICAL MODEL_Mentions"
  - "Solution_HONEYPOT_Mentions"
  - "Solution_SOFTWARE DEFINED NETWORK_Mentions"
  - "Solution_GAME THEORY_Mentions"
  - "Solution_TRAFFIC SHAPING_Mentions"
  - "Solution_PACKET FILTERING_Mentions"
  - "Solution_BLACKHOLING_Mentions"
  - "Solution_RANK CORRELATION_Mentions"
  - "Solution_STRONG AUTHENTICATION_Mentions"
  - "Solution_SSL/TLS_Mentions"
  - "Solution_HTTPS_Mentions"
  - "Solution_CONTINUOUS AUTHENTICATION_Mentions"
  - "Solution_Identity-Based Encryption (IBE)_Mentions"
  - "Solution_DATA SANITIZATION_Mentions"
  - "Solution_OUTLIER DETECTION_Mentions"
  - "Solution_DATA PROVENANCE_Mentions"
  - "Solution_ADVERSARIAL TRAINING_Mentions"
  - "Solution_TRUSTWORTHY AI_Mentions"
  - "Solution_Bayesian Network_Mentions"
  - "Solution_BEHAVIOR BASED DETECTION_Mentions"
  - "Solution_FORMAL VERIFICATION_Mentions"
  - "Solution_SPLIT MANUFACTURING_Mentions"
  - "Solution_VULNERABILITY MANAGEMENT_Mentions"
  - "Solution_FILE INTEGRITY MONITORING_Mentions"
  - "Solution_VPN_Mentions"
  - "Solution_PRIVACY PRESERVING_Mentions"
  - "Solution_SECURE BOOT_Mentions"
  - "Solution_MERKLE SIGNATURE_Mentions"
  - "Solution_LIVENESS DETECTION_Mentions"
  - "Solution_3D FACE RECONSTRUCTION_Mentions"
  - "Solution_BIOMETRICS_Mentions"
  - "Solution_DIGITAL WATERMARK_Mentions"
  - "Solution_APPLICATION WHITELISTING_Mentions"
  - "Solution_DATA BACKUPS_Mentions"
  - "Solution_HIDDEN MARKOV MODEL_Mentions"
  - "Solution_PATCH MANAGEMENT_Mentions"
  - "Solution_DATA AUGMENTATION_Mentions"
  - "Solution_DIMENSIONALITY REDUCTION_Mentions"
  - "Solution_DEFENSIVE DISTILLATION_Mentions"
  - "Solution_RRAM_Mentions"
  - "Solution_SPATIAL SMOOTHING_Mentions"
  - "Solution_NOISE INJECTION_Mentions"
  - "Solution_TAINT ANALYSIS_Mentions"
  - "Solution_NETWORK SEGMENTATION_Mentions"
  - "Solution_USER BEHAVIOR ANALYTICS_Mentions"
  - "Solution_DECEPTION TECHNOLOGY_Mentions"
  - "Solution_RISK ASSESSMENT_Mentions"
  - "Solution_DYNAMIC RESOURCE MANAGEMENT_Mentions"
  - "Solution_SANDBOXING_Mentions"
  - "Solution_DARKNET MONITORING_Mentions"
  - "Solution_VIRTUAL KEYBOARDS_Mentions"
  - "Solution_CODE SIGNING_Mentions"
  - "Solution_PUBLIC KEY INFRASTRUCTURE_Mentions"
  - "Solution_MUTUAL AUTHENTICATION_Mentions"
  - "Solution_ONE TIME PASSWORD_Mentions"
  - "Solution_DYNAMIC BINARY INSTRUMENTATION_Mentions"
  - "Solution_PASSWORD HASHING_Mentions"
  - "Solution_DNSSEC_Mentions"
  - "Solution_CERTIFICATE PINNING_Mentions"
  - "Solution_SECURE SIMPLE PAIRING_Mentions"
  - "Solution_VULNERABILITY ASSESSMENT_Mentions"
  - "Solution_SIEM_Mentions"
  - "Solution_STANDARDIZED COMMUNICATION_Mentions"
  - "Solution_Control Flow Integrity_Mentions"
  - "Solution_VULNERABILITY SCANNER_Mentions"
  - "Solution_PASSWORD STRENGTH METERS_Mentions"
  - "Solution_PASSWORD MANAGEMENT_Mentions"
  - "Solution_PASSWORD POLICY_Mentions"
  - "Solution_GRAPHICAL AUTHENTICATION_Mentions"
  - "Solution_DATA LOSS PREVENTION_Mentions"
  - "Solution_DATA LEAKAGE DETECTION/PREVENTION_Mentions"
  - "Solution_ACTIVITY MONITORING_Mentions"
  - "Solution_MOVING TARGET DEFENSE_Mentions"
  - "Solution_KEYSTROKE DYNAMICS_Mentions"
  - "Solution_ATTACK TREE_Mentions"
  - "Solution_DISTRIBUTED LEDGERS_Mentions"
  - "Solution_SOURCE IDENTIFICATION_Mentions"
  - "Solution_IMAGE RECOGNITION_Mentions"
  - "Solution_HYPERGAME_Mentions"
  - "Solution_NLP/LLM_Mentions"