function [output] = getFeatures(img, Constants)
    output_1 = getGlobalFeatures(img, Constants.net, Constants.layer, Constants.pooling);
    output_2 = getLocalFeatures(img, Constants.net, Constants.layer, Constants.N1, Constants.L*2, Constants.pooling);
    output_3 = getLocalFeatures(img, Constants.net, Constants.layer, Constants.N2, Constants.L, Constants.pooling);
    
    output.global = output_1;
    output.local1 = output_2;
    output.local2 = output_3;
end

