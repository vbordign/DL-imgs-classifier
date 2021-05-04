import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

if not os.path.exists('./figs/'):
    os.makedirs('./figs/')

loss_train, loss_val, acc_val, acc_test, loss_test=torch.load('stats/stats_7.pkl')
f, a = plt.subplots(2,1,figsize=(5,5))
a[0].errorbar(torch.arange(1,num_epochs+1),loss_train.mean(0), loss_train.std(0), linestyle='dashed', capsize =3, marker='.', color='C0')
a[0].errorbar(torch.arange(1,num_epochs+1),loss_val.mean(0), loss_val.std(0), linestyle='dashed', capsize =3, marker='.', color='C2')
a[0].legend(['Training', 'Validation'], fontsize=14)
a[0].set_ylabel('Loss', fontsize=16, labelpad=15)
a[0].set_xlabel('Epoch', fontsize=16)
a[0].set_xlim([0,num_epochs+1])
a[1].errorbar(torch.arange(1,num_epochs+1),acc_val.mean(0), loss_val.std(0), linestyle='dashed', capsize =3, marker='.', color='k')
a[1].set_ylabel('Val. Accuracy (\%)', fontsize=16)
a[1].set_xlabel('Epoch', fontsize=16)
a[1].set_xlim([0,num_epochs+1])
a[0].set_ylim([-.1,.8])
f.tight_layout()
# f.savefig('./figs/train1.pdf')
plt.show()

#%% Write tables
for i in range(12):
    loss_train, loss_val, acc_val, acc_test, loss_test=torch.load(f'./stats/stats_{i}.pkl')

    print(f'${loss_train.mean(0)[-1]:.4f} \pm {loss_train.std(0)[-1]:.4f}$ &$ {loss_val.mean(0)[-1]:.4f} \pm {loss_val.std(0)[-1]:.4f} $&$ {acc_val.mean(0)[-1]:.2f} \pm {acc_val.std(0)[-1]:.2f} $')
loss_train, loss_val, acc_val, acc_test, loss_test = torch.load('./stats/stats_7.pkl')
print(f'Test loss: {loss_test.mean(0):.4f}\pm {loss_test.std(0):.4f}, Test acc: {acc_test.mean(0):.2f}\pm {acc_test.std(0):.2f}')
#%%
for i in range(12):
    loss_train, loss_val, acc_val, acc_test, loss_test = torch.load(f'./stats/stats_ws_{i}.pkl')
    print(
        f'${loss_train.mean(0)[-1]:.4f} \pm {loss_train.std(0)[-1]:.4f}$ &$ {loss_val.mean(0)[-1]:.4f} \pm {loss_val.std(0)[-1]:.4f} $&$ {acc_val.mean(0)[-1]:.2f} \pm {acc_val.std(0)[-1]:.2f} $')
loss_train, loss_val, acc_val, acc_test, loss_test = torch.load('./stats/stats_ws_9.pkl')
print(f'Test loss: {loss_test.mean(0):.4f}\pm {loss_test.std(0):.4f}, Test acc: {acc_test.mean(0):.2f}\pm {acc_test.std(0):.2f}')

#%%
for i in range(12):
    loss_train, loss_val, acc_val, acc_test, loss_test = torch.load(f'./stats/stats_ws_al_{i}.pkl')

    print(
        f'${loss_train.mean(0)[-1]:.4f} \pm {loss_train.std(0)[-1]:.4f}$ &$ {loss_val.mean(0)[-1]:.4f} \pm {loss_val.std(0)[-1]:.4f} $&$ {acc_val.mean(0)[-1]:.2f} \pm {acc_val.std(0)[-1]:.2f} $')

loss_train, loss_val, acc_val, acc_test, loss_test = torch.load('./stats/stats_ws_al_11.pkl')
print(f'Test loss: {loss_test.mean(0):.4f}\pm {loss_test.std(0):.4f}, Test acc: {acc_test.mean(0):.2f}\pm {acc_test.std(0):.2f}')

#%%
