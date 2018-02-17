function [model] = leastSquaresBias(X,y)

% Take bias into consideration
testCal = [(linspace(1,1,size(X,1)))', X]
w = (testCal'*testCal)\testCal'*y

model.w = w;
model.predict = @(model,X) predict(model, [(linspace(1,1,size(X,1)))', X])

end




function [model] = leastSquares(X,y)

% Solve least squares problem (assumes X'*X is invertible)
w = (X'*X)\X'*y;

model.w = w;
model.predict = @predict;

end


function [yhat] = predict(model,Xhat)
w = model.w;
yhat = Xhat*w;
end