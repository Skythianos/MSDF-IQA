function [output] = getLocalFeatures(img, net, layer, N, L, pooling)
    
    [m,n,~] = size(img);
    for i=1:N
        rng(i);
        crop = img(randi(m-L+1)+(0:L-1), randi(n-L+1)+(0:L-1), :);
        if(strcmp(pooling, 'avg'))
            Features(i,:) = globalAveragePooling(activations(net, crop, layer));
        elseif(strcmp(pooling, 'min'))
            Features(i,:) = globalMinimumPooling(activations(net, crop, layer));
        elseif(strcmp(pooling, 'max'))
            Features(i,:) = globalMaximumPooling(activations(net, crop, layer));
        elseif(strcmp(pooling, 'median'))
            Features(i,:) = globalMedianPooling(activations(net, crop, layer));
        else
        
        end
    end
    
    output = [min(Features,[],1), mean(Features,1), max(Features,[],1)];

end

