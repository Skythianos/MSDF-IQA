function [output] = globalMaximumPooling(acts)

    output = zeros(1, size(acts,3));
    for i=1:size(acts,3)
        tmp = acts(:,:,i);
        output(i) = max(tmp(:));
    end

end


