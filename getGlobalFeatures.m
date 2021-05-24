function [output] = getGlobalFeatures(img, net, layer, pooling)
    Acts   = activations(net, img, layer);
    if(strcmp(pooling, 'avg'))
        output = globalAveragePooling(Acts);
    elseif(strcmp(pooling, 'min'))
        output = globalMinimumPooling(Acts);
    elseif(strcmp(pooling, 'max'))
        output = globalMaximumPooling(Acts);
    elseif(strcmp(pooling, 'median'))
        output = globalMedianPooling(Acts);
    else
        
    end
end

