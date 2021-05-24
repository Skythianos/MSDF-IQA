function [output] = globalAveragePooling(acts)

    output = zeros(1, size(acts,3));
    for i=1:size(acts,3)
        tmp = acts(:,:,i);
        output(i) = mean(tmp(:));
    end

end

