%% ============================================================
%  TAREFA 3.1 - Tratamento do Dataset
%  Conhecimento e Raciocínio - 2025/2026
%% ============================================================

%% --- PASSO 1: Carregar o dataset ---
opts = detectImportOptions('dataset_TP.csv');
opts = setvartype(opts, {'maintenance_level','operating_mode', ...
                         'cooling_type','sensor_status','class_cat'}, 'char');
data = readtable('dataset_TP.csv', opts);

fprintf('Dataset carregado: %d linhas, %d colunas\n', height(data), width(data));

%% --- PASSO 2: Converter atributos categóricos para numérico ---
% Os categóricos têm de ser convertidos antes de tratar os missing values,
% para que possamos usar isnan() em todos os atributos de forma uniforme.

% -- maintenance_level (ORDINAL: existe ordem Low < Medium < High) --
% Justificação: é um atributo ordinal - a ordem tem significado
% (mais manutenção = melhor estado do equipamento)
% Codificação: Low=1, Medium=2, High=3
for i = 1:height(data)
    val = strtrim(data.maintenance_level{i});
    switch lower(val)
        case 'low',    data.maintenance_level{i} = '1';
        case 'medium', data.maintenance_level{i} = '2';
        case 'high',   data.maintenance_level{i} = '3';
    end
end
data.maintenance_level = str2double(data.maintenance_level);

% -- operating_mode (ORDINAL: existe ordem Idle < Normal < Overload) --
% Justificação: é um atributo ordinal - representa intensidade crescente
% de operação do equipamento
% Codificação: Idle=1, Normal=2, Overload=3
for i = 1:height(data)
    val = strtrim(data.operating_mode{i});
    switch lower(val)
        case 'idle',     data.operating_mode{i} = '1';
        case 'normal',   data.operating_mode{i} = '2';
        case 'overload', data.operating_mode{i} = '3';
    end
end
data.operating_mode = str2double(data.operating_mode);

% -- cooling_type (BINÁRIO: sem ordem, apenas dois valores) --
% Justificação: atributo binário sem relação de ordem entre Air e Oil
% Codificação: Air=0, Oil=1
for i = 1:height(data)
    val = strtrim(data.cooling_type{i});
    switch lower(val)
        case 'air', data.cooling_type{i} = '0';
        case 'oil', data.cooling_type{i} = '1';
    end
end
data.cooling_type = str2double(data.cooling_type);

% -- sensor_status (BINÁRIO: sem ordem, apenas dois valores) --
% Justificação: atributo binário - OK=funcionamento normal, Warning=aviso
% Codificação: OK=0, Warning=1
for i = 1:height(data)
    val = strtrim(data.sensor_status{i});
    switch lower(val)
        case 'ok',      data.sensor_status{i} = '0';
        case 'warning', data.sensor_status{i} = '1';
    end
end
data.sensor_status = str2double(data.sensor_status);

fprintf('Conversão de categóricos concluída.\n');

%% --- PASSO 3: Preencher missing values nos atributos de entrada ---
% Estratégia por atributo baseada na análise do domínio e outliers:
%
% MÉDIA: atributos com distribuição simétrica e sem/poucos outliers
%   -> temperature (0.0% outliers), vibration (0.0%), current (0.0%),
%      noise_level (0.0%), efficiency (0.0%)
%
% MEDIANA: atributos com outliers identificados (mais robusta)
%   -> rotation_speed (0.9%), voltage (6.6%!), pressure (0.6%),
%      load_val (0.2%), torque (0.1%)
%
% MODA: sempre para atributos categóricos (já convertidos)
%   -> maintenance_level, operating_mode, cooling_type, sensor_status

% -- Atributos preenchidos com MÉDIA --
colsMean = {'temperature', 'vibration', 'current', 'noise_level', 'efficiency'};
for i = 1:length(colsMean)
    col = colsMean{i};
    valores = data.(col);
    idx_falta = isnan(valores);
    valor_medio = mean(valores(~idx_falta));
    valores(idx_falta) = valor_medio;
    data.(col) = valores;
    fprintf('  [MEDIA] %s: %d valores preenchidos com %.4f\n', ...
            col, sum(idx_falta), valor_medio);
end

% -- Atributos preenchidos com MEDIANA --
colsMedian = {'rotation_speed', 'voltage', 'pressure', 'load_val', 'torque'};
for i = 1:length(colsMedian)
    col = colsMedian{i};
    valores = data.(col);
    idx_falta = isnan(valores);
    valor_mediana = median(valores(~idx_falta));
    valores(idx_falta) = valor_mediana;
    data.(col) = valores;
    fprintf('  [MEDIANA] %s: %d valores preenchidos com %.4f\n', ...
            col, sum(idx_falta), valor_mediana);
end

% -- Atributos preenchidos com MODA --
colsMode = {'maintenance_level', 'operating_mode', 'cooling_type', 'sensor_status'};
for i = 1:length(colsMode)
    col = colsMode{i};
    valores = data.(col);
    idx_falta = isnan(valores);
    valor_moda = mode(valores(~idx_falta));
    valores(idx_falta) = valor_moda;
    data.(col) = valores;
    fprintf('  [MODA] %s: %d valores preenchidos com %.0f\n', ...
            col, sum(idx_falta), valor_moda);
end

fprintf('Preenchimento de atributos de entrada concluído.\n');

%% --- PASSO 4: Preencher missing values no TARGET com CBR ---
% O enunciado exige usar CBR (Retrieve):
% Para cada linha sem target, encontrar o caso mais similar
% com target conhecido e copiar a sua classe.

% Pesos para cada um dos 14 atributos de entrada
% Justificação dos pesos:
%   - Peso ALTO (0.10-0.12): atributos diretamente ligados a falhas
%       vibration    -> indicador primário de falha mecânica
%       current      -> indicador primário de falha elétrica
%       voltage      -> fundamental para falhas elétricas
%       temperature  -> sobreaquecimento indica falha
%       torque       -> esforço mecânico excessivo indica falha
%       noise_level  -> ruído súbito indica falha mecânica
%   - Peso MÉDIO (0.06-0.08): atributos de contexto operacional
%       rotation_speed, pressure, efficiency, load_val
%   - Peso BAIXO (0.05): atributos de configuração/contexto
%       maintenance_level, operating_mode, cooling_type, sensor_status
pesos = [0.10, 0.12, 0.07, 0.10, 0.12, ...
         0.07, 0.10, 0.06, 0.06, 0.10, ...
         0.05, 0.05, 0.05, 0.05];
fprintf('Soma dos pesos: %.2f\n', sum(pesos));

% Nomes dos 14 atributos (reutilizados no Passo 5)
nomes = {'temperature','vibration','rotation_speed','voltage','current', ...
         'pressure','noise_level','efficiency','load_val','torque', ...
         'maintenance_level','operating_mode','cooling_type','sensor_status'};

% Construir matriz temporária para o CBR do target
X_cbr = zeros(height(data), 14);
for i = 1:14
    X_cbr(:,i) = data.(nomes{i});
end

% Separar índices com target conhecido e em falta
idx_known   = find(~cellfun(@isempty, data.class_cat));
idx_missing = find(cellfun(@isempty, data.class_cat));

fprintf('\nRegistos com target conhecido: %d\n', length(idx_known));
fprintf('Registos com target em falta:  %d\n', length(idx_missing));

% Calcular min e max apenas com casos conhecidos (para normalização local)
cbr_min = min(X_cbr(idx_known, :));
cbr_max = max(X_cbr(idx_known, :));
cbr_range = cbr_max - cbr_min;
cbr_range(cbr_range == 0) = 1; % evitar divisão por zero

% Para cada caso sem target, procurar o mais similar
fprintf('\nA preencher targets em falta com CBR...\n');
for i = 1:length(idx_missing)
    query = X_cbr(idx_missing(i), :);
    melhor_sim = -1;
    melhor_idx = -1;

    for j = 1:length(idx_known)
        ref = X_cbr(idx_known(j), :);

        % Similaridade local: 1 - distância normalizada
        sim_local = 1 - abs(query - ref) ./ cbr_range;
        sim_local = max(sim_local, 0); % garantir >= 0

        % Similaridade global: média ponderada das similaridades locais
        sim_global = sum(pesos .* sim_local) / sum(pesos);

        if sim_global > melhor_sim
            melhor_sim = sim_global;
            melhor_idx = idx_known(j);
        end
    end

    % Copiar o target do caso mais similar
    data.class_cat{idx_missing(i)} = data.class_cat{melhor_idx};
end

fprintf('Preenchimento do target concluído.\n');

% Verificar que não há mais missing values
fprintf('\n=== VERIFICAÇÃO FINAL ===\n');
for i = 1:width(data)
    col = data.Properties.VariableNames{i};
    if isnumeric(data.(col))
        n_falta = sum(isnan(data.(col)));
    else
        n_falta = sum(cellfun(@isempty, data.(col)));
    end
    if n_falta > 0
        fprintf('ATENÇÃO: %s ainda tem %d valores em falta!\n', col, n_falta);
    end
end
fprintf('Verificação concluída - dataset pronto.\n');

%% --- PASSO 5: Construir matrizes finais e guardar ---

% Nomes dos 14 atributos (usados nas tarefas seguintes)
nomes = {'temperature','vibration','rotation_speed','voltage','current', ...
         'pressure','noise_level','efficiency','load_val','torque', ...
         'maintenance_level','operating_mode','cooling_type','sensor_status'};

% Matriz X com todos os 14 atributos numéricos (já tratados)
X = zeros(height(data), 14);
for i = 1:14
    X(:,i) = data.(nomes{i});
end

% Targets (coluna class_cat)
targets = data.class_cat;

% Calcular min, max e range para normalização min-max
X_min   = min(X);
X_max   = max(X);
X_range = X_max - X_min;
X_range(X_range == 0) = 1; % evitar divisão por zero

% Dataset normalizado (valores entre 0 e 1)
% Nota: o target (class_cat) NÃO é normalizado conforme o enunciado
X_norm = (X - X_min) ./ X_range;

% Guardar CSV para consulta
writetable(data, 'dataset_tratado.csv');
fprintf('Dataset tratado guardado em: dataset_tratado.csv\n');

% Guardar variáveis Matlab para as tarefas seguintes
% Isto evita ter de reprocessar o dataset cada vez
save('dataset_tratado.mat', 'data', 'X', 'X_norm', 'X_min', 'X_max', ...
     'X_range', 'targets', 'nomes', 'pesos');
% save('dataset_tratado.csv','data', 'X', 'X_norm', 'X_min', 'X_max', ...
%      'X_range', 'targets', 'nomes', 'pesos');
fprintf('Variáveis guardadas em: dataset_tratado.mat\n');
fprintf('\nTarefa 3.1 concluída.\n');
