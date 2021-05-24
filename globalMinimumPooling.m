function [output] = globalMinimumPooling(acts)

    output = zeros(1, size(acts,3));
    for i=1:size(acts,3)
        tmp = acts(:,:,i);
        output(i) = min(tmp(:));
    end

end
