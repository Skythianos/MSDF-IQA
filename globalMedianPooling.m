function [output] = globalMedianPooling(acts)

    output = zeros(1, size(acts,3));
    for i=1:size(acts,3)
        tmp = acts(:,:,i);
        output(i) = median(tmp(:));
    end

end

