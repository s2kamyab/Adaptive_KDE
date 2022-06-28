function X = pdf_sample(pdf, N); % sampled data
    
    switch pdf.type
      case 'gmm'
        X = gmm_sample(pdf, N); % sampled data    
        case 'multivariate'
        X = gmm_sample(pdf, N); % sampled data 
      case 'explicit'
        X = explicit_sample(pdf,N);
      otherwise
        error(sprintf('Unknown PDF type: %s', pdf.type));
    end
end