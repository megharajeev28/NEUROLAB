// ============================================================
// NEURO LAB - MAIN
// ============================================================

clear;
clc;
funcprot(0);

exec("neural_network.sci", -1);
exec("callbacks.sci", -1);
exec("gui.sci", -1);

create_neurolab_gui();
