T0 = readtable('owid-covid-data.xlsx');
T2 = readtable('owid-covid-data.xlsx','Sheet','YEAR');

opts = detectImportOptions('owid-covid-data.xlsx');
opts = setvartype(opts,{'VenditaAuto'},'double');

opts2 = detectImportOptions('owid-covid-data.xlsx','Sheet','YEAR');
opts2 = setvartype(opts2,{'AnomalieSulRiscaldamento'},'double');

T = readtable('owid-covid-data.xlsx',opts);
T1 = readtable('owid-covid-data.xlsx',opts2,'Sheet','YEAR');

T.Properties.VariableNames = {'Rif_Mese','Emiss_C02_Carbo','Emiss_C02_GasNa','Emiss_C02_BenAe','Emiss_C02_CoODi','Emiss_C02_LiqId','Emiss_C02_CarJe','Emiss_C02_Keros','Emiss_C02_Lubri','Emiss_C02_BenMo','Emiss_C02_CokPe','Emiss_C02_CoORe','Emiss_C02_AltrP','Emiss_C02_Petro','Emiss_C02_NTotE','Produz_Carbone','Produz_GasNatur','Produz_PetrGreg','Produz_CFosTOT','Produz_Idroelet','Produz_Eolica','Produz_Biomasse','Produz_RinnoTOT','Produz_EnePrTOT','Consum_RinnoTOT','Consum_NRinnTOT','Consum_EnePrTOT','Import_EnePrTOT','Import_PetrOPEC','Consum_CFosTras','CDD','HDD','vendita_auto','Consum_petrolio_Trasp','Consum_Carb_Elettr','Consum_Carb_TOT'};

f1 = figure('Position',[100,100,1250,675])
plot(T.Rif_Mese,T.Emiss_C02_NTotE,"LineWidth",1.3)
xlabel('Temp')
ylabel('Emi')
title('Emi')
hold on
plot(T.Rif_Mese,T.Emiss_C02_Carbo,'Linewidth',1.3)
plot(T.Rif_Mese,T.Emiss_C02_GasNa,'Linewidth',1.3)
plot(T.Rif_Mese,T.Emiss_C02_Petro,'Linewidth',1.3)
ylim([0 700])
legend('Emissioni CO_{2} TOT','Emissioni CO_{2} da Carbone','Emissioni CO_{2} da Gas Naturale','Emissioni CO_{2} da Petrolio')
grid minor
saveas(f1,[pwd '\immagini\01.ConfrontoEmissioniCO2.png'])

f2 = figure('Position',[100,100,1250,675])
plot(T.Rif_Mese,T.Produz_CFosTOT,'LineWidth',1.3)
xlabel('Tem')
ylabel('Pro')
title('Pr')
hold on
plot(T.Rif_Mese,T.Produz_Carbone,'Linewidth',1.3)
plot(T.Rif_Mese,T.Produz_GasNatur,'Linewidth',1.3)
plot(T.Rif_Mese,T.Produz_PetrGreg,'Linewidth',1.3)
plot(T.Rif_Mese,T.Consum_NRinnTOT,'Linewidth',1.3)
legend('Prod. Combustibili Fossili (Non Rinnovabile) TOT','Prod. Carbone','Prod. Gas Naturale','Prod. Petrolio Greggio','Consumo Energia Non Rinnovabile TOT')
grid minor
saveas(f2,[pwd '\immagini\02.ConfrontoP.png'])

f3 = figure('Position',[100,100,1250,675])
plot(T.Rif_Mese,T.Produz_RinnoTOT,'Linewidth',1.3)
xlabel('Te')
ylabel('Pr')
title('Pr')
hold on
plot(T.Rif_Mese,T.Produz_Idroelet,'Linewidth',1.3)
plot(T.Rif_Mese,T.Produz_Eolica,'Linewidth',1.3)
plot(T.Rif_Mese,T.Produz_Biomasse,'Linewidth',1.3)
plot(T.Rif_Mese,T.Consum_RinnoTOT,'Linewidth',1.3)
legend('Prod. Energia Rinnovabile TOT','Prod. Energia Idroelettrica','Prod. Energia Eolica','Prod. Energia da Biomassa','Consumo Energia Rinnovabile TOT')
grid minor
hold off
saveas(f3,[pwd '\immagini\03.ConfrontoPro.png'])

T11 = T([445:end],:);

f4 = figure('Position',[100,100,1250,675])
histfit(T11.Emiss_C02_NTotE, 20,"normal")
title('Dis ')
xlabel('Qua')
ylabel('Fr')
saveas(f4,[pwd '\immagini\04.Dist.png'])

kurtosis(T11.Emiss_C02_NTotE)
skewness(T11.Emiss_C02_NTotE)

[h,p,jbstat,critval] = jbtest(T11.Emiss_C02_NTotE, 0.05)
[h1,p1,jbstat1,critval1] = jbtest(T11.Emiss_C02_NTotE, 0.01)
[h2,p2,jbstat2,critval2] = jbtest(T11.Emiss_C02_NTotE, 0.10)  

[h3,p3,dstat3,critval3] = lillietest(T11.Emiss_C02_NTotE,'Alpha',0.05)

f5 = figure
subplot(1,2,1)
set(f5,'position',[100,100,1250,675]);
scatter(T11.Emiss_C02_NTotE,T11.Produz_Carbone)
h1 = lsline
h1.Color = 'r';
h1.LineWidth = 2;
title('Emi')
xlabel('Qua')
ylabel('Pr')
subplot(1,2,2)
scatter(T11.Emiss_C02_NTotE,T11.Produz_Eolica)
h2 = lsline
h2.Color = 'b';
h2.LineWidth = 2;
title('Emi')
xlabel('Quan')
ylabel('Pro')
saveas(f5,[pwd '\immagini\05.Scatt.png'])

tt = corr(T11{:,{'Emiss_C02_NTotE','Produz_Carbone','Produz_Eolica'}})
rowNames = {'Emissioni CO_2 TOT','Prod_Carbone','Prod_Eolica'};
colNames = {'Emissioni CO_2 TOT','Prod_Carbone','Prod_Eolica'};
sTable = array2table(tt,'RowNames',rowNames,'VariableNames',colNames);

f6 = figure
set(f6,'position',[100,100,1250,675]);
varNames = {'PEoli','PCarb','CRinn','CNRin','CFosTr','CCarE','CCarT','EmC02'};
[R,PValue,H] = corrplot(T11{:,{'Produz_Eolica','Produz_Carbone','Consum_RinnoTOT','Consum_NRinnTOT','Consum_CFosTras','Consum_Carb_Elettr','Consum_Carb_TOT','Emiss_C02_NTotE'}},'varNames',varNames)
saveas(f6,[pwd '\immagini\06.ScatterP.png'])


mhat = fitlm(T11,'ResponseVar','Emiss_C02_NTotE','PredictorVars','Consum_Carb_TOT')

mhat.Coefficients

anova(mhat,'summary')

f7 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, T11.Emiss_C02_NTotE)
hold on
plot(T11.Rif_Mese, mhat.Fitted)
hold off
title('Emi')
xlabel('Te')
ylabel('Qu')
legend('Emis')
saveas(f7,[pwd '\immagini\07.Emission.png'])

mhat.Rsquared
fit1 = mhat.Fitted
res1 = mhat.Residuals.Raw

f8 = figure()
set(f8,'position',[100,100,1250,675]);
subplot(1,2,1)
histfit(res1)
title('Di')
xlabel('Qu')
ylabel('Co')

subplot(1,2,2)
scatter(fit1,res1)
h1 = lsline
h1.Color = 'red';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f8,[pwd '\immagini\08.Residui.png'])

skewness(res1)
kurtosis(res1)

[h,p,jbstat,critval] = jbtest(res1, 0.05)
[h,p,jbstat,critval] = jbtest(res1, 0.01)
[h,p,dstat,critval] = lillietest(res1,'Alpha',0.05)
[h,p,ci,stats] = ttest(res1)

mhat2 = fitlm(T11,'ResponseVar','Emiss_C02_NTotE','PredictorVars',{'Produz_RinnoTOT','Consum_NRinnTOT','Consum_RinnoTOT'})

mhat2.Coefficients

f9a = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, T11.Emiss_C02_NTotE)
hold on
plot(T11.Rif_Mese, mhat2.Fitted)
hold off
title('Ems')
xlabel('Te')
ylabel('Qua')
legend('Emissioni di CO2 dataset','Emissioni di C02 stimate')
saveas(f9a,[pwd '\immagini\09a.Emissioni_realiVS.png'])


anova(mhat2,'summary')

mhat2.Rsquared;
fit2 = mhat2.Fitted;
res2 = mhat2.Residuals.Raw;

f10a = figure()
set(f10a,'position',[100,100,1250,675]);
subplot(1,2,1)
histfit(res2)
title('Di')
xlabel('Qu')
ylabel('Co')

subplot(1,2,2)
scatter(fit2,res2)
h1 = lsline
h1.Color = 'black';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f10a,[pwd '\immagini\10a.Residu.png'])

skewness(res2)
kurtosis(res2)
[h,p,jbstat,critval] = jbtest(res2, 0.05)
[h,p,jbstat,critval] = jbtest(res2, 0.01);
[h,p,dstat,critval] = lillietest(res2,'Alpha',0.05)
[h3,p3,ci3,stats3] = ttest(res2);

mhat9 = fitlm(T11,'ResponseVar','Emiss_C02_NTotE','PredictorVars',{'Produz_Eolica','Produz_Carbone','Consum_CFosTras','Produz_Biomasse','Consum_Carb_TOT'})
mhat9.Coefficients

f9 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, T11.Emiss_C02_NTotE)
hold on
plot(T11.Rif_Mese, mhat9.Fitted)
hold off
title('Em')
xlabel('Te')
ylabel('Qua')
legend('Emissioni di CO2 dataset','Emissioni di C02 stimate')
saveas(f9,[pwd '\immagini\09.Emissio.png'])

anova(mhat9,'summary')

mhat9.Rsquared;

fit9 = mhat9.Fitted;

res9 = mhat9.Residuals.Raw;

f10 = figure()
set(f10,'position',[100,100,1250,675]);
subplot(1,2,1)
histfit(res9)
title('Dis')
xlabel('Qua')
ylabel('Co')

subplot(1,2,2)
scatter(fit9,res9)
h1 = lsline
h1.Color = 'black';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f10,[pwd '\immagini\10.Residu.png'])

skewness(res9)
kurtosis(res9)

[h,p,jbstat,critval] = jbtest(res9, 0.05)
[h,p,jbstat,critval] = jbtest(res9, 0.01)
[h,p,dstat,critval] = lillietest(res9,'Alpha',0.05)
[h3,p3,ci3,stats3] = ttest(res9)

xvars = [{'Produz_Carbone'},{'Produz_GasNatur'},{'Produz_PetrGreg'},{'Produz_CFosTOT'},{'Produz_Idroelet'},{'Produz_Eolica'},{'Produz_Biomasse'},{'Consum_CFosTras'},{'Consum_petrolio_Trasp'},{'Consum_Carb_Elettr'},{'Consum_Carb_TOT'}];
T11_sel = T11(:,[xvars,{'Emiss_C02_NTotE'}]);

X = T11_sel{:,xvars};
y = T11_sel{:,'Emiss_C02_NTotE'};
[b,se,pval,in_stepwise,stats,nextstep,history] = stepwisefit(X,y,...
    'PRemove',0.15,'PEnter',0.05);

mhat_step = fitlm(T11_sel(:,[in_stepwise,true]),'ResponseVar','Emiss_C02_NTotE')

disp('RMSE con stepwise model selection:')
disp(stats.rmse)

f11 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, T11.Emiss_C02_NTotE)
hold on
plot(T11.Rif_Mese, mhat_step.Fitted)
hold off
title('Emss')
xlabel('Te')
ylabel('Qu')
legend('Emissioni di CO2 dataset','Emissioni di C02 stimate')
saveas(f11,[pwd '\immagini\11.Emission.png'])

xvars = [{'Produz_Carbone'},{'Produz_GasNatur'},{'Produz_PetrGreg'},{'Produz_CFosTOT'},{'Produz_Idroelet'},{'Produz_Eolica'},{'Produz_Biomasse'},{'Consum_CFosTras'},{'Consum_petrolio_Trasp'},{'Consum_Carb_Elettr'},{'Consum_Carb_TOT'}];
X = T11{:,xvars};
y = T11.Emiss_C02_NTotE;

[Bhat,lasso_st]=lasso(X,y,'CV',20,'MCReps',5,...
                'Options',statset('UseParallel',true),...
                'PredictorNames',xvars);

lasso_st.IndexMinMSE
in_lasso = not(Bhat(:,lasso_st.IndexMinMSE)==0);

mhat_lasso = fitlm(T11_sel(:,[in_lasso(:)',true]),'ResponseVar','Emiss_C02_NTotE')

lassoPlot(Bhat,lasso_st,'PlotType','CV');
legend('show')

disp('RMSE con 20-folds cross-validation:')
disp(sqrt(lasso_st.MSE(lasso_st.IndexMinMSE)))

f12 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, T11.Emiss_C02_NTotE,'Color',[0.4660, 0.7540, 0.1880],'LineWidth', 1)
hold on
plot(T11.Rif_Mese, mhat_lasso.Fitted,'r','LineWidth', 1)
title('Emss')
xlabel('Te')
ylabel('Qu')
legend('Emissioni di C02 test data','Emissioni di C02 fittati con Lasso')
saveas(f12,[pwd '\immagini\12.Emissio.png'])
LT = trenddecomp(T11.Emiss_C02_NTotE)

f12y = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese,T11.Emiss_C02_NTotE)
hold on
plot(T11.Rif_Mese,LT)
title('Emss')
xlabel('Te')
ylabel('Qu')
legend('Emissioni di C02 test data','Trend')
saveas(f12y,[pwd '\immagini\12y.Emissi.png'])

f13 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(T11.Rif_Mese,T11.Emiss_C02_NTotE);
hold on
plot(T11.Rif_Mese,LT)
title('Se')
subplot(2,2,2)
histfit(T11.Emiss_C02_NTotE,20,'Normal')
title('Is')
subplot(2,2,3)
autocorr(T11.Emiss_C02_NTotE, 48);
title('AC')
subplot(2,2,4)
parcorr(T11.Emiss_C02_NTotE, 48);
title('P')
saveas(f13,[pwd '\immagini\13.ACF_PAC.png'])

[h,pValue,stat,cValue] = lbqtest(T11.Emiss_C02_NTotE,'lags',[6,12,18,24,30,36,42,48])

[pValue,stat] = dwtest(T11.Emiss_C02_NTotE,ones(size(T11.Emiss_C02_NTotE,1)-1,1),'Method','exact')

[h,p,adfstat,critval] = adftest(T11.Emiss_C02_NTotE,'model','TS','lags',0:6)

X = T11_sel(:,xvars);
X_train = X([1:115],:);
X_train_m = table2array(X_train);
X_test = X([116:end],:);
X_test_m = table2array(X_test);

y = T11_sel(:,'Emiss_C02_NTotE');
y_train = y([1:115],:);
y_train_m = table2array(y_train);
y_test = y([116:end],:);
y_test_m = table2array(y_test);

X2 = T11_sel([116:end],:);
periodo = T11.Rif_Mese;
Period_test = periodo([116:end],:);
Period_train = periodo([1:115],:);

AR12 = arima('ARLags',1:12)
EstAR12 = estimate(AR12,y_train_m,'Display','off')
summarize(EstAR12)

innov_tr = infer(EstAR12, y_train_m, 'Y0',y_train_m(1:12));
innov_te = infer(EstAR12, y_test_m, 'Y0',y_test_m(1:12));
new = forecast(EstAR12,24,y_test_m);
fit_right = new+innov_te;

serie_training_ar12 = y_train_m+innov_tr
RMSE = sqrt(mean((y_train_m - serie_training_ar12).^2))

f14 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_train,serie_training_ar12)
legend('Osservata training dataset','Fittata training AR(12)')
xlabel('T')
ylabel('Qu')
title('Valut')

f14 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_test,y_test_m)
plot(Period_test,fit_right)
legend('Osservata training dataset','Osservata test dataset','Fittata AR(12)')
xlabel('T')
ylabel('Qu')
title('Valut')
saveas(f14,[pwd '\immagini\14.Fittin.png'])

RMSE = sqrt(mean((y_test_m - fit_right).^2))

Modello_nuovo = arima(2,2,2);
Modello_nuovo2 = estimate(Modello_nuovo,y_train_m,'Display','off');
summarize(Modello_nuovo2);
innov_nuove = infer(Modello_nuovo2, y_train_m, 'Y0',y_train_m(1:7));
innov_nuove_te2 = infer(Modello_nuovo2, y_test_m, 'Y0',y_test_m(1:7));
nuove_2 = forecast(Modello_nuovo2,24,y_test_m);
fit_right2_nuovi = nuove_2+innov_nuove_te2;

f14A = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_test,y_test_m)
plot(Period_test,fit_right2_nuovi)
legend('Osservata training dataset','Osservata test dataset','Fittata ARIMA(2,3,2)')
xlabel('T')
ylabel('Qu')
title('Valut')saveas(f14A,[pwd '\immagini\14A.Fitting_ARIMA(2,2,2).png'])
RMSE = sqrt(mean((y_test_m - fit_right2_nuovi).^2))

MA11 = arima(2,0,2);
MAS11 = estimate(MA11,y_train_m,'Display','off');
summarize(MAS11);
innovMA112 = infer(MAS11, y_train_m, 'Y0',y_train_m(1:4));
innov_te2 = infer(MAS11, y_test_m, 'Y0',y_test_m(1:4));
new_2 = forecast(MAS11,24,y_test_m);
fit_right2 = new_2+innov_te2;

f15 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_test,y_test_m)
plot(Period_test,fit_right2)
xlabel('Tem')
ylabel('Qua')
legend('Osservata train','Osservata test','Fittata con ARIMA')
title('Ser')
saveas(f15,[pwd '\immagini\15.Fitting_ARIMA.png'])

RMSE = sqrt(mean((y_test_m - fit_right2).^2))

serie_training_arma22 = y_train_m+innovMA112
RMSE = sqrt(mean((y_train_m - serie_training_arma22).^2))

f14 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_train,serie_training_arma22)
legend('Osservata training dataset','Fittata training ARMA(2,2)')
xlabel('Te')
ylabel('Qua')
title('Val')

pMax = 3;
qMax = 3;
AIC = zeros(pMax+1,qMax+1);
BIC = zeros(pMax+1,qMax+1);

for p = 0:pMax
    for q = 0:qMax
        if p == 0 & q == 0
            Mdl = arima(0,0,0);
        end
        if p == 0 & q ~= 0
            Mdl = arima('MALags',1:q,'SARLags',12);
        end
        if p ~= 0 & q == 0
            Mdl = arima('ARLags',1:p,'SARLags',12);
        end
        if p ~= 0 & q ~= 0
            Mdl = arima('ARLags',1:p,'MALags',1:q,'SARLags',12);
        end
        EstMdl = estimate(Mdl,y_train_m,'Display','off');
        results = summarize(EstMdl);
        AIC(p+1,q+1) = results.AIC;
        BIC(p+1,q+1) = results.BIC;
    end
end

minAIC = min(min(AIC))
[bestP_AIC,bestQ_AIC] = find(AIC == minAIC)
bestP_AIC = bestP_AIC - 1; bestQ_AIC = bestQ_AIC - 1;
minBIC = min(min(BIC))
[bestP_BIC,bestQ_BIC] = find(BIC == minBIC)
bestP_BIC = bestP_BIC - 1; bestQ_BIC = bestQ_BIC - 1;
fprintf('%s%d%s%d%s','The model with minimum AIC is SARIMA((', bestP_AIC,',0,',bestQ_AIC,'),(12,0,0))');
fprintf('%s%d%s%d%s','The model with minimum BIC is SARIMA((', bestP_BIC,',0,',bestQ_BIC,'),(12,0,0))');
SARIMA_opt = arima('ARLags',1,'SARLags',12);
summarize(SARIMA_opt)
Est_SARIMA_opt = estimate(SARIMA_opt,y_train_m);
summarize(Est_SARIMA_opt)
E0 = infer(Est_SARIMA_opt, y_train_m, 'Y0',T11.Emiss_C02_NTotE(1:14));
E = infer(Est_SARIMA_opt, y_test_m, 'Y0',T11.Emiss_C02_NTotE(1:14));
fittedSARIMA_opt = y_test_m + E;

RMSE = sqrt(mean((y_test_m - fittedSARIMA_opt).^2))

serie_training_sar = y_train_m+E0
RMSE = sqrt(mean((y_train_m - serie_training_sar).^2))

f14 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_train,serie_training_sar)
legend('Osservata training dataset','Fittata training SAR(12)')
xlabel('Te')
ylabel('Qua')
title('Val')

MA11 = arima(3,0,3);
MAS11 = estimate(MA11,y_train_m,'Display','off');
summarize(MAS11);
innovMA112 = infer(MAS11, y_train_m, 'Y0',y_train_m(1:6));
innov_te9 = infer(MAS11, y_test_m, 'Y0',y_test_m(1:6));
new_2 = forecast(MAS11,24,y_test_m);
fit_right3 = new_2+innov_te9;

f15a = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_test,y_test_m)
plot(Period_test,fit_right2)
xlabel('Te')
ylabel('Quan')
legend('Osservata train','Osservata test','Fittata con ARIMA')
title('Ser')
saveas(f15a,[pwd '\immagini\15a.Fitting_ARIMA(3,0,3).png'])

f16 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese,T11.Emiss_C02_NTotE)
hold on
plot(Period_test,fittedSARIMA_opt)
plot(Period_test,fit_right2)
plot(Period_test,fit_right3)
plot(Period_test,fit_right)
xlabel('Te')
ylabel('Quan')
legend('Osservata','SARIMA((1,0,1),(12,0,0))',...
    'ARIMA(2,0,2)','ARIMA(3,0,3)','AR(12)')
title('S')
saveas(f16,[pwd '\immagini\16.ConfrontoModelli.png'])

media_adf = mean(innov_te2)
innovazioni_normalizzate = innov_te2-media_adf
media_nuova = mean(innovazioni_normalizzate)

f17 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(innovazioni_normalizzate);
hold on
yline(media_nuova,'r')
hold off
title('Se')
subplot(2,2,2)
histfit(innov_te2,10,'Normal')
title('Ist')
subplot(2,2,3)
autocorr(innov_te2);
title('A')
subplot(2,2,4)
parcorr(innov_te2);
title('P')
saveas(f17,[pwd '\immagini\17.ACF_PACF_ARIMA(2,0,2).png'])

[h,p,jbstat,critval] = jbtest(innov_te2)
[h,pValue,stat,cValue] = lbqtest(innov_te2,'lags',[1,6,8,12,13,16])
[h,p,adfstat,critval] = adftest(innovazioni_normalizzate)

[h,pValue,stat,cValue] = archtest(innov_te2)

arma = arima(3,2,1);
aux_arma = estimate(arma,innov_te2);

res_k = infer(aux_arma, innov_te2,'Y0',innov_te2(1:6));

f18 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(res_k)
title('AC')
subplot(2,2,2)
parcorr(res_k)
title('PACi')
saveas(f18,[pwd '\immagini\18.ACF_PACF_residui_modellati.png'])

[h,pValue,stat,cValue] = lbqtest(res_k,'lags',[6,8,12])

pMax = 4;
qMax = 4;
AIC = zeros(pMax+1,qMax+1);
BIC = zeros(pMax+1,qMax+1);

for p = 0:pMax
    for q = 0:qMax
        if p == 0 & q == 0
            Mdl = regARIMA(0,0,0);
        end
        if p == 0 & q ~= 0
            Mdl = regARIMA(0,0,q);
        end
        if p ~= 0 & q == 0
            Mdl = regARIMA(p,0,0);
        end
        if p ~= 0 & q ~= 0
            Mdl = regARIMA(p,0,q);
        end
        EstMdl = estimate(Mdl,y_train_m,'Display','off');
        results = summarize(EstMdl);
        AIC(p+1,q+1) = results.AIC;
        BIC(p+1,q+1) = results.BIC;
    end
end

minAIC = min(AIC(min(AIC>0)))
[bestP_AIC,bestQ_AIC] = find(AIC == minAIC)

minBIC = min(BIC(min(BIC>0)))
[bestP_BIC,bestQ_BIC] = find(BIC == minBIC)

fprintf('%s%d%s%d%s','The model with minimum AIC is ARIMA(', bestP_AIC,',0,',bestQ_AIC,')');
fprintf('%s%d%s%d%s','The model with minimum BIC is ARIMA(', bestP_BIC,',0,',bestQ_BIC,')');

n = size(T11.Emiss_C02_NTotE,1)
s = 12;
sinusoidi = [cos(1*2*pi*[1:n]'/s),...
            cos(2*2*pi*[1:n]'/s),...
            cos(3*2*pi*[1:n]'/s)];

cos_train = sinusoidi([1:115],:);
cos_test = sinusoidi([116:end],:);

RegARIMA1 = regARIMA(2,0,2);
RegARIMA1s = estimate(RegARIMA1, y_train_m,'X',cos_train);
innov_train4 = infer(RegARIMA1s, y_train_m,'X',cos_train);
innov_test4 = infer(RegARIMA1s, y_test_m,'X',cos_test);
[gdpF,gdpMSE] = forecast(RegARIMA1s,24,'Y0',y_test_m);
fit_regARIMA = gdpF+innov_test4;

RMSE = sqrt(mean((y_test_m - fit_regARIMA).^2))

f14b = figure('Position',[100,100,1250,675])
plot(Period_test,y_test_m,'LineWidth', 2)
hold on
plot(Period_test,fittedSARIMA_opt)
plot(Period_test,fit_right2)
plot(Period_test,fit_regARIMA)
xlabel('Tempo [Mesi]')
ylabel('Quantità emessa [Mln di tonnellate]')
legend('Osservata','SARIMA((1,0,1),(12,0,0))',...
    'ARIMA(2,0,2)','regARIMA')
title('Se')
saveas(f14b,[pwd '\immagini\14b.Confron.png'])

serie_training_regARMA = y_train_m+innov_train4
RMSE = sqrt(mean((y_train_m - serie_training_regARMA).^2)) 

f14 = figure('Position',[100,100,1250,675])
plot(Period_train,y_train_m)
hold on
plot(Period_train,serie_training_regARMA)
legend('Osservata training dataset','Fittata training regARMA(2,2)')
xlabel('Tempo [Mesi]')
ylabel('Quantità emessa [Mln di tonnellate]')
title('Va')

f16a = figure('Position',[100,100,1250,675])
plot(Period_test,y_test_m,'LineWidth', 2)
hold on
plot(Period_test,fittedSARIMA_opt)
plot(Period_test,fit_right2)
plot(Period_test,fit_regARIMA)
plot(Period_test,fit_right)
xlabel('Tempo [Mesi]')
ylabel('Quantità emessa [Mln di tonnellate]')
legend('Osservata','SARIMA((1,0,1),(12,0,0))',...
    'ARIMA(2,0,2)','RegARIMA(2,0,2)','AR(12)')
title('Se')
saveas(f16a,[pwd '\immagini\16a.Confro.png'])

LT2 = trenddecomp(T.HDD)
f19a = figure('Position',[100,100,1250,675])
plot(T.Rif_Mese, T.HDD)
hold on
plot(T.Rif_Mese, LT2)
legend('Hdd')
title('Se')
xlabel('Tempo [Mesi]')
ylabel('HDD [Grado giorno]')
saveas(f19a,[pwd '\immagini\19a.Serie_HDD.png'])

yy = T11.Emiss_C02_NTotE;
xx = T11.HDD;
xx2 = xx.^2;
XX = [xx,xx2];
nn = length(yy);

figure
plot(xx,yy,'p')
title('Se')
xlabel('HDD [Grado giorno]')
ylabel('Quantità emessa [Mln di tonnellate]')

lm = fitlm(xx,yy);
lm2 = fitlm(XX,yy);

f19 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, yy)
hold on
plot(T11.Rif_Mese, lm.Fitted)
plot(T11.Rif_Mese, lm2.Fitted)
legend('Emissioni osservate','Emissioni fittate lineari','Emissioni fittate quadratiche')
xlabel('Tempo [Mesi]')
ylabel('Quantità emessa [Mln di tonnellate]')
title('Se')
saveas(f19,[pwd '\immagini\19.Conf.png'])

f20 = figure('Position',[100,100,1250,675])
plot(xx,yy,'p')
title('Se')
xlabel('HD')
ylabel('Qu')
hold on
plot(xx,lm.Fitted,'r','LineWidth',2)
plot(xx,lm2.Fitted,'g','LineWidth',2)
legend('Osservati','Regressione lineari','Regressione quadratica')
saveas(f20,[pwd '\immagini\20.Confro.png'])

tt = corr(T11{:,{'Emiss_C02_NTotE','HDD'}})
rowNames = {'Emissioni CO_2 TOT','HDD'};
colNames = {'Emissioni CO_2 TOT','HDD'};
sTable = array2table(tt,'RowNames',rowNames,'VariableNames',colNames)

tt1=corr(T11.Emiss_C02_NTotE,xx2)

m1 = ssm(@(params)tvp_beta_alphaconstant(params,xx,lm.Coefficients.Estimate(1),lm.Coefficients.Estimate(2)));

params01 = [0.10,log(var(lm.Residuals.Raw))];
disp('Stima')
mhat1 = estimate(m1,yy,params01);

xfilter2 = filter(mhat1,yy);
alpha2.flt = xfilter2(:,1);
figure
plot(alpha2.flt)
beta2.flt = xfilter2(:,2);
figure
plot(beta2.flt)

xsmooth2 = smooth(mhat1,yy);
alpha2.smo = xsmooth2(:,1);
figure
plot(alpha2.smo)
beta2.smo = xsmooth2(:,2);
figure
plot(beta2.smo)

y2_flt = alpha2.flt + beta2.flt.*xx;
e2_flt = yy - y2_flt;
mean(e2_flt)
var(e2_flt)
y2_smo = alpha2.smo + beta2.smo.*xx;
e2_smo = yy - y2_smo;
mean(e2_smo)
var(e2_smo)

f21 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, yy)
hold on
plot(T11.Rif_Mese, y2_flt)
plot(T11.Rif_Mese, y2_smo)
legend('Emissioni osservate','Emissioni filtrate','Emissioni smoothed')
title('Se')
xlabel('HD')
ylabel('Qu')
saveas(f21,[pwd '\immagini\21.Confr.png'])


R2_flt2_alph_const = 1 - mean(e2_flt.^2) / var(yy)
R2_smo2_alph_const = 1 - mean(e2_smo.^2) / var(yy)

m2 = ssm(@(params)tvp_alpha_beta(params,xx,lm.Coefficients.Estimate(1),lm.Coefficients.Estimate(2)));

params0 = [0.10,log(var(lm.Residuals.Raw)),log(var(lm.Residuals.Raw))];

disp('Stima')
mhat2 = estimate(m2,yy,params0);

xfilter3 = filter(mhat2,yy);
alpha3.flt = xfilter3(:,1);
figure
plot(alpha3.flt)

beta3.flt = xfilter3(:,2);
figure
plot(beta3.flt)

xsmooth3 = smooth(mhat2,yy);
alpha3.smo = xsmooth3(:,1);
figure
plot(alpha3.smo)

beta3.smo = xsmooth3(:,2);
figure
plot(beta3.smo)

y3_flt = alpha3.flt + beta3.flt.*xx;
e3_flt = yy - y3_flt;
mean(e3_flt)
var(e3_flt)

y3_smo = alpha3.smo + beta3.smo.*xx;
e3_smo = yy - y3_smo;
mean(e3_smo)
var(e3_smo)

f22 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, yy)
hold on
plot(T11.Rif_Mese, y3_flt)
plot(T11.Rif_Mese, y3_smo)
legend('Emissioni osservate','Emissioni filtrate','Emissioni smoothed')
title('Se')
xlabel('HD')
ylabel('Qu')
saveas(f22,[pwd '\immagini\22.Confron.png'])

R2_stat = lm.Rsquared.Adjusted
R2_stat_QUADR = lm2.Rsquared.Adjusted
R2_flt2 = 1 - mean(e3_flt.^2) / var(yy)
R2_smo2 = 1 - mean(e3_smo.^2) / var(yy)

f23 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, yy)
hold on
plot(T11.Rif_Mese, lm2.Fitted)
plot(T11.Rif_Mese, y2_flt)
legend('Emissioni osservate','Emiss modello quadratico','Emissioni filtrata')
title('Se')
xlabel('HD')
ylabel('Qu')
saveas(f23,[pwd '\immagini\23.Confro.png'])

primi_parte = lm2.Fitted
primi_parte1 = primi_parte([1:115],:);
secod_parte = y3_flt([116:end],:);

t5_comp = [primi_parte1;secod_parte]
e5_comp = yy - t5_comp;
mean(e5_comp)
var(e5_comp)

R2_composizione = 1 - mean(e5_comp.^2) / var(yy)

f23t = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese, yy)
hold on
plot(T11.Rif_Mese, t5_comp)
legend('Emissioni osservate','Emissioni modello ibrido')
title('Se')
xlabel('HD')
ylabel('Qu')

lm2.Rsquared;

fit_quadrat_hdd = lm2.Fitted;

res_quadrat_hdd = lm2.Residuals.Raw;

f24 = figure('Position',[100,100,1250,675])
set(f24,'position',[100,100,1250,675]);
subplot(1,2,1)
histfit(res_quadrat_hdd)
title('Se')
xlabel('HD')
ylabel('Qu')

subplot(1,2,2)
scatter(fit_quadrat_hdd,res_quadrat_hdd)
h1 = lsline
h1.Color = 'black';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f24,[pwd '\immagini\24.Resid.png'])

skewness(res_quadrat_hdd)
kurtosis(res_quadrat_hdd)

[h,p,jbstat,critval] = jbtest(res_quadrat_hdd, 0.05)
[h,p,jbstat,critval] = jbtest(res_quadrat_hdd, 0.01)
[h,p,dstat,critval] = lillietest(res_quadrat_hdd,'Alpha',0.05)

[h,pValue,stat,cValue] = archtest(res_quadrat_hdd)


res_quadrat_hdd2 = res_quadrat_hdd.^2;
f25 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(res_quadrat_hdd,24)
title('AC')
subplot(2,2,2)
parcorr(res_quadrat_hdd,24)
title('PA')
subplot(2,2,3)
autocorr(res_quadrat_hdd2,24)
title('AC')
subplot(2,2,4)
parcorr(res_quadrat_hdd2,24)
title('PA')
saveas(f25,[pwd '\immagini\25.Autoc.png'])

arma2 = arima(3,0,5);
aux_arma2 = estimate(arma2,res_quadrat_hdd);

res_k2 = infer(aux_arma2, res_quadrat_hdd,'Y0',res_quadrat_hdd(1:8));

res2_k2 = res_k2.^2;

f26 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(res_k2)
title('AC')
subplot(2,2,2)
parcorr(res_k2)
title('PA')
subplot(2,2,3)
autocorr(res2_k2,24)
title('AC')
subplot(2,2,4)
parcorr(res2_k2,24)
title('PA')
saveas(f26,[pwd '\immagini\26.Autocor.png'])

m0 = garch(0,2)
[mhat,covM,logL] = estimate(m0,res2_k2);
condVhat = infer(mhat,res2_k2);
condVol = sqrt(condVhat);
n = 139;
[a,b] = aicbic(logL,mhat.P+mhat.Q,n)
f27 = figure('Position',[100,100,1250,675])
plot(T11.Rif_Mese,res2_k2)
hold on;
plot(T11.Rif_Mese,condVol)
title('Re')
xlabel('Tem')
legend('Quantità emessa','Estim. cond. volatility','Location','NorthEast')
hold off;
saveas(f27,[pwd '\immagini\27.Garch(0,2).png'])

std_res = res2_k2 ./ condVol;
std_res2 = std_res .^ 2;

f27a = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(std_res)
title('St')
subplot(2,2,2)
histogram(std_res,10)
subplot(2,2,3)
autocorr(std_res)
subplot(2,2,4)
parcorr(std_res)
saveas(f27a,[pwd '\immagini\27a.resid.png'])

f27b = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(std_res2)
title('Sta')
subplot(2,2,2)
histogram(std_res2,10)
subplot(2,2,3)
autocorr(std_res2)
subplot(2,2,4)
parcorr(std_res2)
saveas(f27b,[pwd '\immagini\27b.resid.png'])

tt3=corr(T1.AnomalieSulRiscaldamento, T1.TotalEnergyCO2EmissionsUSA)

mhat = fitlm(T1,'ResponseVar','AnomalieSulRiscaldamento','PredictorVars','TotalEnergyCO2EmissionsUSA')

mhat.Coefficients

anova(mhat,'summary')

mhat.Rsquared

fit70 = mhat.Fitted

res70 = mhat.Residuals.Raw

f28 = figure('Position',[100,100,1250,675])
plot(T1.Years,T1.AnomalieSulRiscaldamento)
hold on
plot(T1.Years, mhat.Fitted)
hold off
title('Se')
xlabel('HD')
ylabel('Qu')
legend('Anomalie Riscaldamento dataset','Anomalie Riscaldamento stimati')
saveas(f28,[pwd '\immagini\28.Confron.png'])

f29 = figure('Position',[100,100,1250,675])
subplot(1,2,1)
histfit(res70)
title('Se')
xlabel('HD')
ylabel('Qu')

subplot(1,2,2)
scatter(fit70,res70)
h1 = lsline
h1.Color = 'black';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f29,[pwd '\immagini\29.Resi.png'])

skewness(res70)
kurtosis(res70)
[h,p,jbstat,critval] = jbtest(res70, 0.05)
[h,p,jbstat,critval] = jbtest(res70, 0.01)
[h,p,dstat,critval] = lillietest(res70,'Alpha',0.05)
[h3,p3,ci3,stats3] = ttest(res70)

mhat = fitlm(T1,'ResponseVar','AnomalieSulRiscaldamento','PredictorVars',{'TotalEnergyCO2EmissionsUSA','TotalEnergyCO2EmissionChina','TotalEnergyCO2EmissionRussia'})
mhat.Coefficients
mhat.Rsquared

fit71 = mhat.Fitted
res71 = mhat.Residuals.Raw

anova(mhat,'summary')

f30 = figure('Position',[100,100,1250,675])
plot(T1.Years,T1.AnomalieSulRiscaldamento)
hold on
plot(T1.Years, mhat.Fitted)
hold off
title('Se')
xlabel('HD')
ylabel('Qu')
legend('Anomalie Riscaldamento dataset','Anomalie Riscaldamento stimati')
saveas(f30,[pwd '\immagini\30.Confron.png'])

f31 = figure('Position',[100,100,1250,675])
subplot(1,2,1)
histfit(res71)
title('Se')
xlabel('HD')
ylabel('Qu')

subplot(1,2,2)
scatter(fit71,res71)
h1 = lsline
h1.Color = 'black';
h1.LineWidth = 2;
xlabel('Valori fittati');
ylabel('Residui di regressione');
saveas(f31,[pwd '\immagini\31.Res.png'])

skewness(res71)
kurtosis(res71)
[h,p,jbstat,critval] = jbtest(res71, 0.05)
[h,p,jbstat,critval] = jbtest(res71, 0.01)
[h,p,dstat,critval] = lillietest(res71,'Alpha',0.05)
[h3,p3,ci3,stats3] = ttest(res71)

[h,pValue,stat,cValue] = lbqtest(res71,'lags',[1,4,6])

f50 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(res71)
title('AC')
subplot(2,2,2)
parcorr(res71)
title('PAC')

f32 = figure('Position',[100,100,1250,675])
plot(T1.Years,T1.TotalEnergyCO2EmissionsUSA,"LineWidth",1.3)
title('Se')
xlabel('HD')
ylabel('Qu')
hold on
plot(T1.Years,T1.TotalEnergyCO2EmissionChina,'Linewidth',1.3)
plot(T1.Years,T1.TotalEnergyCO2EmissionRussia,'Linewidth',1.3)
legend('Emissioni CO_{2} USA','Emissioni CO_{2} Cina','Emissioni CO_{2} Russia')
grid minor
saveas(f32,[pwd '\immagini\32.Con.png'])

res_opt4 = regARIMA('ARLags',1)
mod_reg_ARMA4 = estimate(res_opt4, res71);
innov_ARMA4 = infer(mod_reg_ARMA4, res71);

f33 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(innov_ARMA4)
title('A')
subplot(2,2,2)
parcorr(innov_ARMA4)
title('PA')
saveas(f33,[pwd '\immagini\33.RegA.png'])

res_opt = regARIMA('Intercept',5,'AR',{0.1 0.2},'MA',{0.5},...
    'Variance',0.5)

mod_reg_ARMA = estimate(res_opt, res71);
innov_ARMA = infer(mod_reg_ARMA, res71);
summarize(mod_reg_ARMA)

f34 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
autocorr(innov_ARMA)
title('A')
subplot(2,2,2)
parcorr(innov_ARMA)
title('PA')
saveas(f34,[pwd '\immagini\34.RegARIMA(2,0,1)_anomalie.png'])
