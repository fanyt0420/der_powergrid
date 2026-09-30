"""Generate the standard IEEE 13-node feeder plus phase-unbalanced DER QSTS data."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
from topology_plot import plot_network_topology

ROOT=Path(__file__).resolve().parents[2]; CASE=ROOT/'data'/'ieee13_unbalanced_der'; BASE=CASE/'ieee13_base'; H=list(range(24))
URL='https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases'
LOADS=[('671',[1,2,3]),('634a',[1]),('634b',[2]),('634c',[3]),('645',[2]),('646',[2,3]),('692',[1,3]),('675a',[1]),('675b',[2]),('675c',[3]),('611',[3]),('652',[1]),('670a',[1]),('670b',[2]),('670c',[3])]
BUS={'650':[1,2,3],'RG60':[1,2,3],'632':[1,2,3],'633':[1,2,3],'634':[1,2,3],'645':[2,3],'646':[2,3],'670':[1,2,3],'671':[1,2,3],'680':[1,2,3],'692':[1,2,3],'675':[1,2,3],'684':[1,3],'611':[3],'652':[1]}
EDGES=[('650632','RG60','632'),('632670','632','670'),('670671','670','671'),('671680','671','680'),('632633','632','633'),('632645','632','645'),('645646','645','646'),('671692','671','692'),('692675','692','675'),('671684','671','684'),('684611','684','611'),('684652','684','652')]
POS={'650':(0,0),'RG60':(1,0),'632':(2,0),'670':(3,0),'671':(4,0),'680':(5,0),'633':(3,2),'634':(4,2),'645':(3,-2),'646':(4,-2),'692':(5,-1),'675':(6,-1),'684':(5,2),'611':(6,2),'652':(6,3)}
def main():
    CASE.mkdir(parents=True,exist_ok=True); BASE.mkdir(exist_ok=True); prof=CASE/'der_profiles'; prof.mkdir(exist_ok=True)
    master=BASE/'IEEE13Nodeckt.dss'; codes=BASE/'IEEELineCodes.DSS'
    if not master.exists(): urlretrieve(URL+'/13Bus/IEEE13Nodeckt.dss',master)
    if not codes.exists(): urlretrieve(URL+'/IEEELineCodes.DSS',codes)
    # Coordinates are supplied by our generated topology PNG, so avoid an
    # optional OpenDSS GUI coordinate-file dependency.
    master.write_text(master.read_text(encoding='utf-8').replace('BusCoords IEEE13Node_BusXY.csv', '! Bus coordinates omitted; see network_topology.png'), encoding='utf-8')
    # 12 DERs distributed over trunk, low-voltage transformer secondary, and
    # single/two/three-phase laterals.  They total 550 kW inverter generation,
    # 210 kW storage power, and 150 kW controllable EV demand.
    ders=[
      {'id':'PV_632_A','kind':'pv','bus':'632','phases':[1],'p_rated_kw':70}, {'id':'PV_670_C','kind':'pv','bus':'670','phases':[3],'p_rated_kw':50},
      {'id':'PV_671_A','kind':'pv','bus':'671','phases':[1],'p_rated_kw':80}, {'id':'PV_675_A','kind':'pv','bus':'675','phases':[1],'p_rated_kw':100},
      {'id':'PV_675_C','kind':'pv','bus':'675','phases':[3],'p_rated_kw':70}, {'id':'PV_634_ABC','kind':'pv','bus':'634','phases':[1,2,3],'p_rated_kw':180},
      {'id':'BESS_671_B','kind':'bess','bus':'671','phases':[2],'p_rated_kw':100,'energy_kwh':400,'reserve_soc':20,'initial_soc':55},
      {'id':'BESS_645_B','kind':'bess','bus':'645','phases':[2],'p_rated_kw':60,'energy_kwh':240,'reserve_soc':20,'initial_soc':60},
      {'id':'BESS_684_A','kind':'bess','bus':'684','phases':[1],'p_rated_kw':50,'energy_kwh':200,'reserve_soc':20,'initial_soc':50},
      {'id':'EV_671_C','kind':'ev','bus':'671','phases':[3],'p_rated_kw':60}, {'id':'EV_675_B','kind':'ev','bus':'675','phases':[2],'p_rated_kw':50},
      {'id':'EV_611_C','kind':'ev','bus':'611','phases':[3],'p_rated_kw':40}, {'id':'Wind_680_ABC','kind':'wind','bus':'680','phases':[1,2,3],'p_rated_kw':120},
    ]
    for d in ders: d['profile_file']=f"der_profiles/{d['id'].lower()}.csv"
    devices=[{'id':n,'kind':'load','bus':next(b for b in BUS if n.startswith(b)),'phases':p} for n,p in LOADS]+ders
    net={'base':{'frequency_hz':60,'base_kv_ll':4.16,'slack_bus':'650'},'simulation':{'start_datetime':'2026-01-01 00:00:00','end_datetime':'2026-01-02 00:00:00','step_seconds':3600},'buses':[{'id':b,'phases':p,'is_slack':b=='650'} for b,p in BUS.items()],'branches':[{'id':i,'from_bus':a,'to_bus':b,'phases':BUS[a] if len(BUS[a])<=len(BUS[b]) else BUS[b],'kind':'line','length_km':0.1} for i,a,b in EDGES],'devices':devices}
    (CASE/'network.json').write_text(json.dumps(net,indent=2),encoding='utf-8')
    dss=[f'Redirect "{master.resolve()}"']
    for d in ders:
      phases='.'.join(map(str,d['phases'])); n=len(d['phases']); kv='.48' if d['bus']=='634' else '2.4'
      if d['kind'] in {'pv','wind'}: dss.append(f"New Generator.{d['id']} phases={n} bus1={d['bus']}.{phases} kV={kv} kW=0 kvar=0")
      elif d['kind']=='ev': dss.append(f"New Load.{d['id']} phases={n} bus1={d['bus']}.{phases} conn=wye kV={kv} kW=0 kvar=0")
      else: dss.append(f"New Storage.{d['id']} phases={n} bus1={d['bus']}.{phases} kV={kv} kWrated={d['p_rated_kw']} kWhrated={d['energy_kwh']} %stored={d['initial_soc']} %reserve={d['reserve_soc']} dispmode=EXTERNAL kW=0 kvar=0")
    (CASE/'network.dss').write_text('\n'.join(dss+['Solve','']),encoding='utf-8')
    shape=[.55,.52,.5,.5,.53,.62,.72,.82,.9,.96,1,1.02,1,1.01,1.05,1.1,1.18,1.25,1.3,1.28,1.15,.95,.75,.65]
    base={'671':(1155,660),'634a':(160,110),'634b':(120,90),'634c':(120,90),'645':(170,125),'646':(230,132),'692':(170,151),'675a':(485,190),'675b':(68,60),'675c':(290,212),'611':(170,80),'652':(128,86),'670a':(17,10),'670b':(66,38),'670c':(117,68)}
    rows=[]
    for h in H:
      for n,_ in LOADS:
       p,q=base[n]; factor=shape[h]*(1+{'675a':.08,'675b':-.05,'675c':.03}.get(n,0)); rows.append({'timestamp':f'2026-01-01 {h:02d}:00:00','load_name':n,'p_kw':round(p*factor,3),'q_kvar':round(q*factor,3)})
    pd.DataFrame(rows).to_csv(CASE/'load_profiles.csv',index=False)
    ts=[f'2026-01-01 {h:02d}:00:00' for h in H]
    for i,d in enumerate(ders):
      file=prof/f"{d['id'].lower()}.csv"; r=d['p_rated_kw']
      if d['kind']=='pv': pd.DataFrame({'timestamp':ts,'pv_kw':[max(0,r*(1-abs(h-(12+i%3))/6)) for h in H],'pv_kvar':0,'controller_type':'volt_var','q_limit_kvar':r*.35,'v_ref_pu':1,'droop_kvar_per_pu':r*5}).to_csv(file,index=False)
      elif d['kind']=='wind': pd.DataFrame({'timestamp':ts,'wind_kw':[r*(.45+.45*((h+i)%6)/5) for h in H],'wind_kvar':0,'controller_type':'constant_pq'}).to_csv(file,index=False)
      elif d['kind']=='bess': pd.DataFrame({'timestamp':ts,'bess_kw':[-.6*r if 10<=h<=15 else .5*r if 18<=h<=21 else 0 for h in H],'bess_kvar':0,'controller_type':'soc_schedule'}).to_csv(file,index=False)
      else: pd.DataFrame({'timestamp':ts,'ev_kw':[r if 18<=h<=22 else 0 for h in H],'ev_kvar':0,'controller_type':'constant_pq'}).to_csv(file,index=False)
    plot_network_topology(net,CASE/'network_topology.png',POS); print(CASE)
if __name__=='__main__': main()
