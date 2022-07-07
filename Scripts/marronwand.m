function mw = marronwand()
    mw = cell(15,1);
            
    
    mw{1}.means =  [0];
    mw{1}.stds  =  [1];
    mw{1}.priors = [1];

    
    mw{2}.means =  [0 1/2 13/12];
    mw{2}.stds  =  [1 (2/3) (5/9)];
    mw{2}.priors = [1 1 3]/5;
    
    
    mw{3}.means =  3*((2/3).^(0:7) - 1);
    mw{3}.stds  =  (2/3).^(0:7);
    mw{3}.priors = ones(1,8)/8;
    
    
    mw{4}.means =  [0  0];
    mw{4}.stds  =  [1  1/10];
    mw{4}.priors = [2 1]/3;
    
    
    mw{5}.means =  [0  0];
    mw{5}.stds  =  [1  1/10];
    mw{5}.priors = [1 9]/10;
    
    
    mw{6}.means =  [-1  1];
    mw{6}.stds  =  [2/3 2/3];
    mw{6}.priors = [1 1]/2;
    
    
    mw{7}.means =  [-3/2  3/2];
    mw{7}.stds  =  [1/2   1/2];
    mw{7}.priors = [1  1]/2;
    
    
    mw{8}.means =  [0  3/2];
    mw{8}.stds  =  [1  1/3] ;
    mw{8}.priors = [3  1]/4;
    
    
    mw{9}.means =  [-6/5  6/5  0];
    mw{9}.stds  =  [3/5  3/5  1/4];
    mw{9}.priors = [9 9 2]/20;
    
    
    mw{10}.means =  [0 (0:4)/2-1 ];
    mw{10}.stds  =  [1 ones(1,5)/10];
    mw{10}.priors = [5 ones(1,5)]/10;
    
    
    mw{11}.means =  [-1 1 ((0:6)-3)/2];
    mw{11}.stds  =  [2/3 2/3 ones(1,7)/100];
    mw{11}.priors = [49/100 49/100 ones(1,7)/350];
    
    
    mw{12}.means =  [0   ((-2:2)+1/2)];
    mw{12}.stds  =  [1   (2.^(-(-2:2)))/10];
    mw{12}.priors = [1/2 (2.^(1-(-2:2)))/31];
    
    
    mw{13}.means =  [-1  1   -(1:3)/2 (1:3)/2];
    mw{13}.stds  =  [2/3 2/3 ones(1,3)/100 ones(1,3)*7/100];
    mw{13}.priors = [46/100  46/100 ones(1,3)/300 ones(1,3)*7/300];
    
    
    mw{14}.means =  [(65-96*(.5.^(0:5)))/21];
    mw{14}.stds  =  [(32/63)./(2.^(0:5))];
    mw{14}.priors = [2.^(5-(0:5))/63 ];
    
    
    mw{15}.means =  [(12*(0:2)-15)/7  2*(8:10)/7];
    mw{15}.stds  =  [6 6 6 1 1 1]/21;
    mw{15}.priors = [6 6 6 1 1 1]/21;

    
    
    
end