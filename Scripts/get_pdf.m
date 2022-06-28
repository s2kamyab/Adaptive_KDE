function pdf = get_pdf(no)
  
  if no == 0
    pdf.type = 'gmm';
    pdf.means = [-4 3];
    pdf.stds  = [3  1];
    pdf.priors = [1 1]/2;
    
  elseif no <= 15
    mw = marronwand();
    pdf = mw{no};
    pdf.type = 'gmm';
    
  elseif no == 16
    pdf.type = 'explicit';
    pdf.Z = 0:.01:1;
    pdf.P = 6 * (pdf.Z - pdf.Z.^2);
  elseif no <=18
      pdf.type = 'multivariate';
      pdf.means = [2 2;-2 -2]';
      pdf.stds  = [3  1];
      pdf.priors = [1 1]/2;
      
  end
  
  
  
end