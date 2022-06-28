function P = pdf_eval(pdf, Z)
    
    switch pdf.type
     case 'gmm'
       P = gmm_eval(pdf, Z);
      case 'explicit'
        
        
       P = interp1(pdf.Z, pdf.P, Z, 'cubic', 0); 
       
       
     otherwise
       error(sprintf('Unknown PDF type: %s', pdf.type));
    end
    
end