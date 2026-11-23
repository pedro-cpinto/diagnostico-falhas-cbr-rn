%% ============================================================
%  MAIN - Executa todas as tarefas por ordem
%% ============================================================

clc; clear; close all;

fprintf('=== TAREFA 3.1 - Tratamento do Dataset ===\n');
run('tarefa3_1.m');

fprintf('\n=== TAREFA 3.2 - Sistema CBR ===\n');
run('tarefa3_2_cbr.m');

fprintf('\n=== TAREFA 3.3 - Redes Neuronais ===\n');
run('tarefa3_3_rn_TESTE.m');
