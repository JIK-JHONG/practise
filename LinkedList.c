#include <stdio.h>
#include <stdlib.h>

typedef struct LinkedList
{
    int data;
    struct LinkedList *next;
} Nodes;

Nodes *init()
{
    return (Nodes *)malloc(sizeof(Nodes));
}
Nodes *find(Nodes *target, int position)
{
    int idx = 0;
    Nodes *current = target;
    while (current != NULL && idx < position)
    {
        current = current->next;
        idx++;
    }
    return current;
}
int nodes_calc(Nodes *target)
{
    int idx = 0;
    Nodes *current = target;
    while (current != NULL)
    {
        current = current->next;
        idx++;
    }
    return idx;
}
void modified_node(Nodes **target,int data, int position){
    int total_node = nodes_calc(*target);
    if (position <= total_node){
        Nodes *mod = find(*target, position-1);
        mod->data = data;
    }else{
        printf("無法修改：超出範圍(Total Nodes=%d)，顯示位置需要 < %d\n",total_node,total_node);
    }    
}
void remove_node(Nodes **target,int position){
    int total_node = nodes_calc(*target);
    if(position == 0 && *target !=NULL){
        Nodes *temp = *target;
        *target = (*target)->next;
        free(temp);    
    }else if (position <= total_node){
        Nodes *prev = find(*target, position-1);
        Nodes *temp = prev->next;
        prev->next = prev->next->next;
        free(temp);
    }else{
        printf("無法刪除：超出範圍(Total Nodes=%d)，顯示位置需要 < %d\n",total_node,total_node);
    }    

}
void insert(Nodes **target, int data)
{
    if (*target == NULL)
    {
        *target = init();
        (*target)->data = data;
        (*target)->next = NULL;
    }
    else
    {
        Nodes *current = *target;
        while (current->next != NULL)
        {
            current = current->next;
        }
        Nodes *new_node = init();
        new_node->data = data;
        new_node->next = NULL;
        current->next = new_node;
    }
}
void insert_node(Nodes **target, int data,int type,int position){
    if(type == 0 ){
        Nodes *head = init();
        head->data = data;
        head->next = *target;    
        *target = head;    
    }else if(type == -1){
        insert(target,data);
    }else{
        int total_node = nodes_calc(*target);
        if (position <= total_node){
            Nodes *prev_node = find(*target, position -1);
            Nodes *new_node = init();
            new_node->data = data;
            new_node->next = prev_node->next;
            prev_node->next = new_node;
        }else{
            printf("Position 超過鏈表長度，將數據插入到尾端\n");
            insert(target, data);
        }
    }

}
void show_node(Nodes *target, int position)
{
    int total_node = nodes_calc(target);
    if (position <= total_node){
        Nodes *node = find(target, position -1);
        printf("------\n");
        printf("[Val] = %d\n", node->data);
        printf("[Nex] = %p\n", node->next);
        printf("[Mem] = %p\n", node);
        printf("------\n");
    }else{
        printf("超出範圍(Total Nodes=%d)，顯示位置需要 < %d\n",total_node,total_node);
    }
}
void show_travel(Nodes *target)
{
    printf("------\n");
    while (target != NULL)
    {
        printf("[Val] = %d\n", target->data);
        printf("[Nex] = %p\n", target->next);
        printf("[Mem] = %p\n", target);
        target = target->next;
    }
    printf("------\n");
}

int main()
{
    int data_set[] = {0,1,2,3,4};
    int N = sizeof(data_set) / sizeof(int);
    Nodes *node = NULL;
    for (int i=0;i<N;i++){
        insert_node(&node, data_set[i],-1,0);
    }
    modified_node(&node,10,2);
    // insert_node(&node, 2,0,0);
    // insert_node(&node, 4,1,0);
    show_travel(node);
    show_node(node,5);

    return 0;
}