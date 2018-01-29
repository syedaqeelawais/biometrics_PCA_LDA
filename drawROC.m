
function [far, frr] = roc(genf,impf,type,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      [far,frr] = roc(genf,impf, type, varargin)
%%      Function to compute ROC (not optimized)
%%      gen = all genuine scores in a file (column vector)
%%      imp = all impostor scores in a file (column vector)
%%      type = 'd' - distance scores; 's' - similarity scores
%%      Author: Arun Ross
%%      Last Modified: Oct 2006, Feb 2014 SC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Begin ROC..');

gfid = fopen(genf, 'r');
gen = fscanf(gfid, '%f');

iid = fopen(impf, 'r');
imp = fscanf(iid, '%f');

%Determine range of scores
TotGen = length(gen);
TotImp = length(imp);
MinScore = min(min(gen),min(imp));
MaxScore = max(max(gen),max(imp));
Inc = (MaxScore-MinScore)/100;

%Determine histogram of genuine and impostor scores
hgen = histc(gen, MinScore-Inc:Inc:MaxScore+Inc);
himp = histc(imp, MinScore-Inc:Inc:MaxScore+Inc);
figure; plot(MinScore-Inc:Inc:MaxScore+Inc,hgen,'b'); hold on; plot(MinScore-Inc:Inc:MaxScore+Inc,himp,'r');
legend('Genuine Distribution', 'Impostor Distribution');

%The cumulative helps in computing frr/grr at various thresholds
frr = cumsum(hgen);
grr = cumsum(himp);
    
frr = frr/TotGen*100;
grr = grr/TotImp*100;
far = 100 - grr;
gar = 100 - frr;

%Invert definition of far/frr/gar/grr if the scores are distance measures 
if (type=='d')
    far = 100 - far;
    gar = 100 - gar;
    frr = 100 - frr;
    grr = 100 - grr;
end

%Plot ROC
figure
plot(far, frr, varargin{:})
xlabel('False Accept Rate (%)', 'FontSize',14);
ylabel('False Reject Rate (%)', 'FontSize',14);
set(gca,'FontSize',14);

disp('End ROC..');

%Plot EER Line
hold on; plot(0:100, 0:100, 'k--');
grid on;
legend('ROC Curve', 'EER Line', 'Location', 'Best');