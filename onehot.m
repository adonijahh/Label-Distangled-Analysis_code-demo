function one_hot = onehot(Y)
if min(size(Y,2))==1
    I=eye(max(Y));
    one_hot=I(Y,:);
    if size(one_hot,2) ~= 6
       one_hot = [one_hot,zeros(size(one_hot,1),6-size(one_hot,2))]; 
    end
else
    one_hot = double(categorical(Y)); 
end