import os

def run_cmd(cmd):
    ret = os.system(cmd)
    if ret != 0:
        raise Exception(f'cmd failed:\n{cmd}')

for sigmaV_aux_alpha in [1,3,10]:
    #sigmaV_aux_alpha *= 0.1
    with open('a','w') as handler:
        handler.write(str(sigmaV_aux_alpha))

    run_cmd(f'rm -rf outs/LP_vtoeye3_rand{sigmaV_aux_alpha}')
    run_cmd(f'cp -r outs/LP_vtoeye3 outs/LP_vtoeye3_rand{sigmaV_aux_alpha}')
    #run_cmd(f'rm -f outs/LP_vtoeye3_rand{sigmaV_aux_alpha}/V_aux.pt')
    run_cmd(f'python del_lambda.py LP_vtoeye3_rand{sigmaV_aux_alpha} all')
    run_cmd(f'python run_list.py LP_vtoeye3_rand{sigmaV_aux_alpha}')